#include "GalaxyFactory.h"
#include "BarnesHutParticleSystem.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>


GalaxyFactory::GalaxyFactory(BarnesHutParticleSystem& particle_system)
    : particle_system_(particle_system) {
}

// Helper functions for mass profile calculations
double GalaxyFactory::hernquist_Menc(double r, double Mb, double a) {
    // r, a in kpc; Mb in 1e9 Msun
    return Mb * (r*r) / ((r + a) * (r + a));
}

double GalaxyFactory::disk_exp_Menc(double r, double Md, double Rd) {
    // Thin exponential disk cumulative mass (2D): Md [1 - e^{-r/Rd}(1 + r/Rd)]
    const double x = r / Rd;
    return Md * (1.0 - std::exp(-x) * (1.0 + x));
}

double GalaxyFactory::nfw_Menc(double r, double Mh, double rvir, double c) {
    // Normalized NFW: Mh * f(x)/f(c), x=r/rs, rs=rvir/c
    const double rs = rvir / c;
    const auto f = [](double y){ return std::log(1.0 + y) - y/(1.0 + y); };
    const double x  = r / rs;
    const double fc = f(c);
    return (fc > 0.0) ? Mh * (f(x) / fc) : 0.0;
}

double GalaxyFactory::vcirc_code(double r_kpc, double Menc_1e9Msun) {
    // v_code = sqrt(G * M / (r + eps)) in km/s, then divide by VELOCITY_UNIT_KMS
    const double v_kms = std::sqrt(G_GALACTIC * std::max(0.0, Menc_1e9Msun)
                                   / (r_kpc + DEFAULT_SOFTENING_KPC));
    return v_kms / VELOCITY_UNIT_KMS;
}

double GalaxyFactory::clamp_min_r(double r, double rmin) { 
    return (r < rmin) ? rmin : r; 
}

void GalaxyFactory::create_realistic_milky_way() {
    std::cout << "\n🌌 Creating CANONICAL Milky Way - debugging missing chunk...\n";
    particle_system_.clear_particles();

    std::mt19937 gen(std::random_device{}());
    const double scale = 0.01;
    
    const double disk_scale_length = 3.5 / scale;      
    const double bulge_radius = 0.6 / scale;           
    const double halo_scale_radius = 200.0 / scale;    
    const double max_radius = 25.0 / scale;            
    
    const double smbh_mass = 0.0043 / scale;           
    const double bulge_mass = 20.0 / scale;            
    const double disk_mass = 55.0 / scale;             
    const double halo_mass = 1200.0 / scale;           
    
    std::uniform_real_distribution<double> angle_dist(0, 2 * M_PI);
    std::normal_distribution<double> velocity_noise(0.0, 0.02);
    
    // Debug: Track particle distribution by angle
    std::vector<int> angle_bins(36, 0);  // 10-degree bins
    auto track_angle = [&](double x, double y) {
        double angle = std::atan2(y, x);
        if (angle < 0) angle += 2 * M_PI;
        int bin = (int)(angle * 18.0 / M_PI);  // Convert to 0-35 range
        if (bin >= 0 && bin < 36) angle_bins[bin]++;
    };
    
    struct Spawned {
        float x, y, vx, vy;
        double m;
        Vec3 color;
    };
    std::vector<Spawned> spawned;
    
    double Px = 0, Py = 0, Mtot = 0;

    // Central SMBH
    spawned.push_back({0, 0, 0, 0, (float)smbh_mass, {1.0f, 1.0f, 0.8f}});
    Mtot += smbh_mass;

    // BULGE - Check for issues in Hernquist sampling
    int bulge_particles = 3000;
    std::cout << "Creating bulge with Hernquist profile...\n";
    
    int failed_bulge = 0;
    for (int i = 0; i < bulge_particles; ++i) {
        double u = std::max(1e-12, std::uniform_real_distribution<double>(0.0, 1.0)(gen));
        
        // ISSUE 1: Check Hernquist inverse CDF
        double sqrt_u = std::sqrt(u);
        if (sqrt_u >= 1.0) {
            failed_bulge++;
            continue;  // Skip invalid particles
        }
        
        double r = bulge_radius * sqrt_u / (1.0 - sqrt_u);
        r = std::min(std::max(r, 2.0 * DEFAULT_SOFTENING_KPC), max_radius);
        
        double th = angle_dist(gen);
        double x = r * std::cos(th);
        double y = r * std::sin(th);
        
        track_angle(x, y);

        double Menc = smbh_mass
                    + hernquist_Menc(r, bulge_mass, bulge_radius)
                    + disk_exp_Menc(r, disk_mass, disk_scale_length)
                    + nfw_Menc(r, halo_mass, halo_scale_radius, 10.0);

        double v = vcirc_code(r, Menc);
        double vx = -v * std::sin(th) + velocity_noise(gen);
        double vy = v * std::cos(th) + velocity_noise(gen);

        double m = bulge_mass / double(bulge_particles);
        Px += m * vx; Py += m * vy; Mtot += m;
        spawned.push_back({(float)x, (float)y, (float)vx, (float)vy, (float)m, {1.0f, 0.8f, 0.4f}});
    }
    
    if (failed_bulge > 0) {
        std::cout << "⚠️  " << failed_bulge << " bulge particles failed generation\n";
    }

    // DISK - More robust exponential disk generation  
    int disk_particles = 80000;
    std::cout << "Creating disk with exponential profile...\n";
    
    // ISSUE 2: Replace the problematic Gamma(2, Rd) with simpler exponential
    std::exponential_distribution<double> exp_dist(1.0 / disk_scale_length);
    
    int failed_disk = 0;
    for (int i = 0; i < disk_particles; ++i) {
        // FIXED: Use single exponential instead of sum of two
        double r = exp_dist(gen);
        
        // Apply reasonable limits
        r = std::min(r, 4.0 * disk_scale_length);  // Max ~14 kpc
        r = std::max(r, 2.0 * DEFAULT_SOFTENING_KPC);  // Min softening
        
        // ISSUE 3: Check if angle distribution is uniform
        double th = angle_dist(gen);
        
        // Verify angle is in valid range
        if (th < 0 || th >= 2 * M_PI) {
            failed_disk++;
            th = std::fmod(th, 2 * M_PI);
            if (th < 0) th += 2 * M_PI;
        }
        
        double x = r * std::cos(th);
        double y = r * std::sin(th);
        
        // ISSUE 4: Check for NaN/inf values
        if (!std::isfinite(x) || !std::isfinite(y)) {
            std::cout << "⚠️  Invalid disk particle: r=" << r << ", th=" << th 
                      << ", x=" << x << ", y=" << y << "\n";
            failed_disk++;
            continue;
        }
        
        track_angle(x, y);

        double Menc = smbh_mass
                    + hernquist_Menc(r, bulge_mass, bulge_radius)
                    + disk_exp_Menc(r, disk_mass, disk_scale_length)
                    + nfw_Menc(r, halo_mass, halo_scale_radius, 10.0);

        double v = vcirc_code(r, Menc);
        
        // Check velocity is reasonable
        if (!std::isfinite(v) || v > 1000.0) {  // Sanity check
            std::cout << "⚠️  Extreme velocity: v=" << v << " at r=" << r << "\n";
            v = std::min(v, 100.0);  // Cap velocity
        }
        
        double vx = -v * std::sin(th) + velocity_noise(gen);
        double vy = v * std::cos(th) + velocity_noise(gen);

        double m = disk_mass / double(disk_particles);
        Px += m * vx; Py += m * vy; Mtot += m;
        spawned.push_back({(float)x, (float)y, (float)vx, (float)vy, (float)m, {0.7f, 0.6f, 1.0f}});
    }
    
    if (failed_disk > 0) {
        std::cout << "⚠️  " << failed_disk << " disk particles had issues\n";
    }

    // HALO - Simplified NFW generation
    int halo_particles = 40000;
    std::cout << "Creating halo with NFW profile...\n";
    
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double c_halo = 10.0;
    const double rs = halo_scale_radius / c_halo;
    
    int failed_halo = 0;
    for (int i = 0; i < halo_particles; ++i) {
        double r = 0.0;
        
        // ISSUE 5: Simplified rejection sampling for NFW
        bool accepted = false;
        for (int attempt = 0; attempt < 50; ++attempt) {  // Limit attempts
            r = uniform(gen) * max_radius;
            double x_nfw = r / rs;
            
            if (x_nfw <= 0) continue;  // Avoid division by zero
            
            double f_nfw = std::log(1.0 + x_nfw) - x_nfw / (1.0 + x_nfw);
            double f_max = std::log(1.0 + max_radius / rs);
            
            if (f_max <= 0) break;  // Avoid division by zero
            
            if (uniform(gen) <= f_nfw / f_max) {
                accepted = true;
                break;
            }
        }
        
        if (!accepted) {
            r = uniform(gen) * max_radius;  // Fallback to uniform
            failed_halo++;
        }
        
        r = std::max(r, 2.0 * DEFAULT_SOFTENING_KPC);
        double th = angle_dist(gen);
        
        // Add 3D projection for more realistic halo
        double phi = std::acos(2.0 * uniform(gen) - 1.0);  // Isotropic
        double x = r * std::sin(phi) * std::cos(th);
        double y = r * std::sin(phi) * std::sin(th);
        
        track_angle(x, y);

        double sigma_kms = 100.0 / std::sqrt(1.0 + r / rs);
        double sigma_code = sigma_kms / VELOCITY_UNIT_KMS;
        std::normal_distribution<double> vel_disp(0.0, sigma_code);
        
        double vx = vel_disp(gen);
        double vy = vel_disp(gen);

        double m = halo_mass / double(halo_particles);
        Px += m * vx; Py += m * vy; Mtot += m;
        spawned.push_back({(float)x, (float)y, (float)vx, (float)vy, (float)m, {0.2f, 0.0f, 0.4f}});
    }
    
    if (failed_halo > 0) {
        std::cout << "⚠️  " << failed_halo << " halo particles used fallback sampling\n";
    }

    // DEBUG: Print angle distribution
    std::cout << "\n🔍 ANGLE DISTRIBUTION ANALYSIS:\n";
    for (int i = 0; i < 36; ++i) {
        double angle_deg = i * 10.0;
        std::cout << "  " << angle_deg << "°-" << (angle_deg + 10) << "°: " << angle_bins[i] << " particles";
        if (angle_bins[i] < 100) {  // Flag suspiciously low bins
            std::cout << " ⚠️  LOW";
        }
        std::cout << "\n";
    }
    
    // Find minimum and maximum bins
    auto min_bin = *std::min_element(angle_bins.begin(), angle_bins.end());
    auto max_bin = *std::max_element(angle_bins.begin(), angle_bins.end());
    std::cout << "  Range: " << min_bin << " to " << max_bin << " particles per 10° sector\n";
    
    if (max_bin > 2 * min_bin) {
        std::cout << "⚠️  UNEVEN DISTRIBUTION DETECTED! Some angles have significantly fewer particles.\n";
    }

    // Center of mass correction
    double vx_cm = Px / Mtot;
    double vy_cm = Py / Mtot;
    
    for (auto& p : spawned) {
        float adj_vx = p.vx - (float)vx_cm;
        float adj_vy = p.vy - (float)vy_cm;
        particle_system_.add_particle(
            Vec2{p.x, p.y}, Vec2{adj_vx, adj_vy}, p.m, p.color
        );
    }

    // Configure physics and camera for galaxy scale
    configure_for_galaxy_scale();

    std::cout << "✅ MW created: " << spawned.size() << " particles\n";
    
    // ISSUE 6: Also check if it's a rendering/buffer issue
    size_t actual_count = particle_system_.get_particle_count();
    std::cout << "   Particle system reports: " << actual_count << " particles\n";
    
    if (actual_count != spawned.size()) {
        std::cout << "⚠️  PARTICLE COUNT MISMATCH! Generated " << spawned.size() 
                  << " but system has " << actual_count << "\n";
    }
}

void GalaxyFactory::create_andromeda_like_galaxy() {
    std::cout << "\n🌌 Creating Andromeda-like (M31) galaxy...\n";
    particle_system_.clear_particles();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // M31 is more massive than Milky Way
    const float total_mass = 1.8e12f;  // Solar masses (scaled)
    const float disk_mass = 8.0e10f;   // Larger stellar disk
    const float bulge_mass = 4.0e10f;  // Larger central bulge
    const float halo_mass = total_mass - disk_mass - bulge_mass;
    
    const float disk_scale_length = 5.0f;  // Larger than MW
    const float disk_scale_height = 0.4f;  
    const float bulge_radius = 1.2f;       // Larger bulge
    const float virial_radius = 30.0f;     // Larger halo
    
    std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
    std::normal_distribution<float> velocity_noise(0.0f, 0.8f);
    
    // Central SMBH (larger than Milky Way's)
    particle_system_.add_particle(Vec2(0, 0), Vec2(0, 0), 1.4e8f / 1e9f, Vec3(1.0f, 0.9f, 0.3f));
    
    // Larger, more massive bulge
    int bulge_particles = 3000;
    std::gamma_distribution<float> bulge_radius_dist(2.0f, bulge_radius / 2.0f);
    
    for (int i = 0; i < bulge_particles; ++i) {
        float r = std::min(bulge_radius_dist(gen), bulge_radius * 3.0f);
        float theta = angle_dist(gen);
        float x = r * cos(theta);
        float y = r * sin(theta);
        
        float v_circ = sqrt(80.0f / (r + 0.1f));
        float vx = -v_circ * sin(theta) + velocity_noise(gen);
        float vy = v_circ * cos(theta) + velocity_noise(gen);
        
        Vec3 color(1.0f, 0.7f, 0.3f);  // More reddish bulge
        particle_system_.add_particle(Vec2(x, y), Vec2(vx, vy), 
                                    bulge_mass / bulge_particles / 1e9f, color);
    }
    
    // Extended stellar disk
    int disk_particles = 18000;
    std::exponential_distribution<float> disk_radius_dist(1.0f / disk_scale_length);
    std::normal_distribution<float> height_dist(0.0f, disk_scale_height);
    
    for (int i = 0; i < disk_particles; ++i) {
        float r = std::min(disk_radius_dist(gen), disk_scale_length * 6.0f);
        float theta = angle_dist(gen);
        
        float x = r * cos(theta);
        float y = r * sin(theta);
        
        // M31 rotation curve (higher velocities)
        float v_circ = 26.0f * sqrt(r) / sqrt(r + 1.5f) + 20.0f;
        if (r > 10.0f) v_circ = 26.0f;
        
        float vx = -v_circ * sin(theta) + velocity_noise(gen);
        float vy = v_circ * cos(theta) + velocity_noise(gen);
        
        Vec3 color(0.9f, 0.7f, 1.0f);  // Bluer stars
        particle_system_.add_particle(Vec2(x, y), Vec2(vx, vy), 
                                    disk_mass / disk_particles / 1e9f, color);
    }
    
    // Larger dark matter halo
    int halo_particles = 10000;
    const float concentration = 10.0f;
    const float scale_radius = virial_radius / concentration;
    
    std::uniform_real_distribution<float> uniform(0, 1);
    std::uniform_real_distribution<float> halo_radius_dist(0, virial_radius);
    
    for (int i = 0; i < halo_particles; ++i) {
        float r, nfw_prob;
        do {
            r = halo_radius_dist(gen);
            float x = r / scale_radius;
            nfw_prob = 1.0f / (x * (1.0f + x) * (1.0f + x));
        } while (uniform(gen) > nfw_prob * scale_radius / r);
        
        float theta = angle_dist(gen);
        float phi = acos(2 * uniform(gen) - 1);
        
        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        
        std::normal_distribution<float> vel_disp(0.0f, 10.0f * sqrt(1.0f / (1.0f + r / scale_radius)));
        float vx = vel_disp(gen);
        float vy = vel_disp(gen);
        
        Vec3 color(0.4f, 0.1f, 0.6f);
        particle_system_.add_particle(Vec2(x, y), Vec2(vx, vy), 
                                    halo_mass / halo_particles / 1e9f, color);
    }
    
    configure_for_galaxy_scale();
}


static inline double vcirc_plummer_code(double r_kpc, double M_1e9Msun, double eps_kpc) {
    double r2 = r_kpc * r_kpc;
    double a2 = eps_kpc * eps_kpc;
    // v^2 = G M r^2 / (r^2 + eps^2)^(3/2)
    double v_kms = std::sqrt( (G_GALACTIC * M_1e9Msun * r2) / std::pow(r2 + a2, 1.5) );
    return v_kms / VELOCITY_UNIT_KMS;
}

// GalaxyFactory.cpp
void GalaxyFactory::create_bh_ring_galaxy(
    const Vec2& center, const Vec2& bulk_vel,
    double bh_mass_1e9Msun, int star_count,
    float r_inner_kpc, float r_outer_kpc,
    float core_eps_kpc, bool clockwise)
{
    // Represent the SMBH as a Plummer core (just a point mass; softening is in the velocity law)
    particle_system_.add_particle(center, bulk_vel, (float)bh_mass_1e9Msun, Vec3(1.0f, 0.95f, 0.75f));

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::uniform_real_distribution<float> TH(0.0f, 2.0f * (float)M_PI);

    auto sample_radius = [&](float r0, float r1){
        float u = U(rng);
        return std::sqrt(r0*r0 + u * (r1*r1 - r0*r0));  // PDF ∝ r
    };

    const float rmin = std::max(r_inner_kpc, core_eps_kpc * 2.5f);  // keep stars well outside the core
    const float rmax = std::max(r_outer_kpc, rmin + 1.0f);
    const float m_star = 1e-6f;

    for (int i = 0; i < star_count; ++i) {
        float r  = sample_radius(rmin, rmax);
        float th = TH(rng);

        float x = r * std::cos(th), y = r * std::sin(th);

        float tx = clockwise ?  std::sin(th) : -std::sin(th);
        float ty = clockwise ? -std::cos(th) :  std::cos(th);

        double v_code = vcirc_plummer_code(r, bh_mass_1e9Msun, core_eps_kpc);

        Vec2 pos = center + Vec2(x, y);
        Vec2 vel = bulk_vel + Vec2((float)(v_code * tx), (float)(v_code * ty));

        particle_system_.add_particle(pos, vel, m_star, Vec3(0.82f, 0.70f, 1.0f));
    }
}




void GalaxyFactory::create_two_spiral_binary_merger_100k() {
        std::cout << "\n💫 BH + ring merger (stability locked, MW scale, ≈100k)\n";
    particle_system_.clear_particles();

    // Same display scaling as your Milky Way builder
    const double scale = 0.01;
    const double S     = 1.0 / scale;   // positions ×100

    // ---------------- Stability knobs ----------------
    const double eps_kpc       = 8.0;        // soft core for SMBH (gentle center)
    const double Rin_kpc       = 5.0 * eps_kpc; // keep stars well outside core
    const double Rout_kpc      = 1.4 * Rin_kpc; // narrow ring band (≈56 kpc if eps=8)
    const double r_mid_kpc     = 0.5 * (Rin_kpc + Rout_kpc);

    const double v_target_code = 6.0;        // code-units at r_mid (safe + visibly rotating)
    const double v_cap_code    = 10.0;       // hard cap to prevent numeric blow-ups

    // Binary orbit: avoid a center-punch; first pass wider than the ring
    const double r0_kpc        = 300.0;
    const double rp_kpc        = std::max(1.5 * Rout_kpc, 90.0);

    // Convert to world units
    const double r0  = r0_kpc * S;
    const double rp  = rp_kpc * S;
    const float  Rin = float(Rin_kpc  * S);
    const float  Rout= float(Rout_kpc * S);
    const double eps = eps_kpc;             // keep eps in kpc for the speed law

    // Solve MBH so vcirc(r_mid) == v_target_code (in code units)
    const double v_target_kms = v_target_code * VELOCITY_UNIT_KMS;
    const double MBH_phys_1e9 = (v_target_kms*v_target_kms) * (r_mid_kpc + eps_kpc) / G_GALACTIC;

    // Match your MW scaling convention for masses too
    const double MBH1 = MBH_phys_1e9 * S;
    const double MBH2 = MBH_phys_1e9 * S;
    const double Mtot = MBH1 + MBH2;

    // Parabolic approach using that total mass
    const double mu = G_GALACTIC * Mtot;
    const double h  = std::sqrt(2.0 * mu * rp);
    const double v  = std::sqrt(2.0 * mu / r0);
    const double vt = h / r0;
    const double vr = -std::sqrt(std::max(0.0, v*v - vt*vt));
    const Vec2 vrel((float)(vr / VELOCITY_UNIT_KMS), (float)(vt / VELOCITY_UNIT_KMS));

    const double f1 = MBH2 / Mtot, f2 = MBH1 / Mtot;
    Vec2 cA(-(float)(0.5 * r0), 0.0f), cB((float)(0.5 * r0), 0.0f);
    Vec2 vA(-f1 * vrel.x, -f1 * vrel.y), vB(f2 * vrel.x, f2 * vrel.y);

    // ------- Builder (inlined) with a velocity cap & narrow ring sampling -------
    const int   N      = 50000;
    const float m_star = 1e-8f; // even lighter: reduce self-gravity noise

    auto clamp = [](double x, double lo, double hi){ return std::max(lo, std::min(x, hi)); };

    auto build_ring = [&](const Vec2& C, const Vec2& V, double MBH, bool cw) {
        particle_system_.add_particle(C, V, (float)MBH, Vec3(1.0f, 0.95f, 0.75f)); // central SMBH

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        std::uniform_real_distribution<float> TH(0.0f, 2.0f * (float)M_PI);

        // Box–Muller for a tight band around r_mid
        auto normal01 = [&](){
            float u1 = std::max(1e-7f, U(rng));
            float u2 = U(rng);
            return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * (float)M_PI * u2);
        };

        const float r_mid = float(r_mid_kpc * S);
        const float sigma = 0.08f * r_mid; // ~8% radial thickness

        for (int i = 0; i < N; ++i) {
            float th = TH(rng);

            // sample a narrow ring and clamp into [Rin, Rout]
            float r  = r_mid + sigma * (float)normal01();
            r = std::max(std::min(r, Rout), Rin);

            float x = r * std::cos(th), y = r * std::sin(th);

            // circular speed with soft core, in *code* units
            double r_kpc = double(r) / S;
            double v_code = std::sqrt(G_GALACTIC * MBH / (r_kpc + eps_kpc)) / VELOCITY_UNIT_KMS;

            // safety cap
            if (v_code > v_cap_code) v_code = v_cap_code;

            float tx = cw ?  std::sin(th) : -std::sin(th);
            float ty = cw ? -std::cos(th) :  std::cos(th);

            Vec2 pos = C + Vec2(x, y);
            Vec2 vel = V + Vec2((float)(v_code * tx), (float)(v_code * ty));

            particle_system_.add_particle(pos, vel, m_star, Vec3(0.82f, 0.70f, 1.0f));
        }
    };

    build_ring(cA, vA, MBH1, /*clockwise=*/false);
    build_ring(cB, vB, MBH2, /*clockwise=*/true);


    // Huge sandbox so nothing hits the walls if something gets energetic
    PhysicsConfig cfg;
    cfg.boundary_min_x = -800.0f * (float)S;  cfg.boundary_max_x =  800.0f * (float)S;
    cfg.boundary_min_y = -600.0f * (float)S;  cfg.boundary_max_y =  600.0f * (float)S;
    cfg.bounce_force   = 0.0f;
    cfg.damping        = 1.0f;
    cfg.time_scale     = 0.4f; // smaller dt → far more stable near pericenter
    if (on_update_physics_params) on_update_physics_params(cfg);
    if (on_update_camera) on_update_camera(0.0, 0.0, 240.0 * S);

    std::cout << "✅ Total: " << particle_system_.get_particle_count()
              << " | MBH(each)=" << MBH1 << " (×1e9 Msun units) | v_cap=" << v_cap_code << " code\n";
}





void GalaxyFactory::create_galaxy_merger_scenario() {
    std::cout << "\n💥 Creating galaxy merger scenario...\n";
    particle_system_.clear_particles();
    
    // Create two galaxies that will merge
    create_milky_way_at_position(Vec2(-15.0f, 0.0f), Vec2(2.0f, 1.0f), 1.0f, "MW-like");
    create_smaller_galaxy_at_position(Vec2(50.0f, 5.0f), Vec2(-3.0f, -0.5f), 0.6f, "Satellite");
    
    configure_for_local_group_scale();
    std::cout << "✅ Galaxy merger scenario created\n";
    std::cout << "   Two galaxies on collision course\n";
    std::cout << "   Watch as tidal forces create streams and eventually merge!\n";
}

void GalaxyFactory::create_local_group_simulation() {
    std::cout << "\n🌌 Creating Local Group simulation...\n";
    particle_system_.clear_particles();
    
    // Milky Way
    create_milky_way_at_position(Vec2(-20.0f, 0.0f), Vec2(0.5f, 0.0f), 1.0f, "Milky Way");
    
    // Andromeda (approaching for collision in ~4.5 Gyr)
    create_andromeda_at_position(Vec2(30.0f, -10.0f), Vec2(-1.2f, 0.8f), 1.5f, "Andromeda");
    
    // Large Magellanic Cloud
    create_smaller_galaxy_at_position(Vec2(-18.0f, -3.0f), Vec2(1.0f, 0.2f), 0.1f, "LMC");
    
    // Small Magellanic Cloud  
    create_smaller_galaxy_at_position(Vec2(-19.0f, -4.0f), Vec2(0.8f, 0.1f), 0.05f, "SMC");
    
    configure_for_local_group_scale();
    std::cout << "✅ Local Group simulation created\n";
    std::cout << "   Milky Way, Andromeda, and satellite galaxies\n";
    std::cout << "   Realistic masses and trajectories\n";
}

void GalaxyFactory::create_twin_spiral_pair() {
    std::cout << "\n🌌 Spawning twin-armed spiral *pair*\n";
    particle_system_.clear_particles();

    // Separation & masses consistent with the builder above
    const double scale = 0.01;
    const double D = 80.0 / scale;              // kpc separation
    const double Mgal = (0.0043 + 20.0 + 55.0 + 1200.0) / scale; // 1e9 Msun total (rough)
    const double G = G_GALACTIC;                // keep your code's G
    const double v_rel_kms = std::sqrt(G * (2.0*Mgal) / D); // two-body circular
    const double v_code    = v_rel_kms / VELOCITY_UNIT_KMS;

    // Place galaxies at +/- D/2 along x; give opposite y-velocities
    Vec2 c1(-float(0.5*D), 0.0f), c2(float(0.5*D), 0.0f);
    Vec2 v1(0.0f,  float( 0.5*v_code));
    Vec2 v2(0.0f,  float(-0.5*v_code));

    // Build each galaxy; flip handedness for visual variety
    create_two_armed_spiral_galaxy(c1, v1, 1.0, 16.0, 60000, 3000, 25000, 0.18, 0.0, false);
    create_two_armed_spiral_galaxy(c2, v2, 1.0, 16.0, 60000, 3000, 25000, 0.18, M_PI*0.35, true);


    PhysicsConfig config;
    config.boundary_min_x = -150.0f;
    config.boundary_max_x = 150.0f;
    config.boundary_min_y = -150.0f;
    config.boundary_max_y = 150.0f;
    config.bounce_force = 0.0f;
    if (on_update_physics_params) on_update_physics_params(config);

    std::cout << "✅ Twin spirals created: " << particle_system_.get_particle_count() << " particles\n";
}

void GalaxyFactory::create_dwarf_galaxy_accretion() {
    std::cout << "\n🔥 Creating dwarf galaxy accretion event...\n";
    particle_system_.clear_particles();
    
    // Main spiral galaxy
    create_milky_way_at_position(Vec2(0.0f, 0.0f), Vec2(0.0f, 0.0f), 1.0f, "Host Galaxy");
    
    // Multiple dwarf galaxies being accreted
    std::vector<Vec2> positions = {
        Vec2(12.0f, 8.0f), Vec2(-15.0f, -6.0f), Vec2(18.0f, -12.0f),
        Vec2(-10.0f, 14.0f), Vec2(22.0f, 3.0f)
    };
    std::vector<Vec2> velocities = {
        Vec2(-1.5f, -1.0f), Vec2(1.8f, 0.5f), Vec2(-2.0f, 1.2f),
        Vec2(0.8f, -1.8f), Vec2(-1.8f, -0.3f)
    };
    
    for (size_t i = 0; i < positions.size(); ++i) {
        create_smaller_galaxy_at_position(positions[i], velocities[i], 0.08f, "Dwarf");
    }
    
    configure_for_galaxy_scale();
    std::cout << "✅ Dwarf galaxy accretion scenario created\n";
    std::cout << "   Multiple small galaxies falling into main halo\n";
}

void GalaxyFactory::create_tidally_disrupting_flyby() {
    std::cout << "\n🌀 Creating *tidal flyby* (A tears B apart)\n";
    particle_system_.clear_particles();

    // --- Choose galaxy models already in your code ---
    // Primary: Andromeda-like (heavier), Satellite: compact dwarf
    const float s1 = 1.5f;   // primary mass scale (Andromeda helper)
    const float s2 = 0.30f;  // satellite mass scale (dwarf helper)

    // These masses match what the helpers actually spawn (in 1e9 Msun units):
    // Andromeda helper ~ (BH 0.14 + bulge 40 + disk 80) * s1
    const double M1 = (0.14 + 40.0 + 80.0) * s1;  // -> ~180 for s1=1.5
    // Dwarf helper ~ (BH 0.001 + disk 10) * s2
    const double M2 = (0.001 + 10.0) * s2;        // -> ~3.0 for s2=0.30
    const double Mtot = M1 + M2;

    // --- Orbit design (parabolic, e=1) in physical units ---
    // Distances in kpc, velocities in km/s then converted to your code units.
    const double r0 = 120.0;  // initial separation (kpc)
    const double rp = 18.0;   // desired pericentre (kpc) — close enough to shred
    const double mu = G_GALACTIC * Mtot; // km^2/s^2·kpc per (1e9 Msun)

    // Parabolic relations:
    // specific angular momentum h = sqrt(2 mu rp)
    // speed at radius r0: v = sqrt(2 mu / r0)
    // tangential component at r0: vt = h / r0
    // radial component (inward):   vr = -sqrt(max(0, v^2 - vt^2))
    const double h  = std::sqrt(2.0 * mu * rp);
    const double v  = std::sqrt(2.0 * mu / r0);
    const double vt = h / r0;
    const double vr = -std::sqrt(std::max(0.0, v*v - vt*vt));

    // Convert to your simulation velocity units (divide by VELOCITY_UNIT_KMS)
    const double vt_code = vt / VELOCITY_UNIT_KMS;
    const double vr_code = vr / VELOCITY_UNIT_KMS;

    // Place galaxies along x; give relative velocity (vr along -x, vt along +y)
    const float xA = (float)(-0.5 * r0), yA = 0.0f;
    const float xB = (float)( 0.5 * r0), yB = 0.0f;

    // Split relative velocity by masses so COM stays at rest
    const double f1 =  M2 / Mtot;
    const double f2 =  M1 / Mtot;
    const Vec2 vrel((float)vr_code, (float)vt_code);

    Vec2 vA(-f1 * vrel.x, -f1 * vrel.y);  // primary moves opposite to B
    Vec2 vB( f2 * vrel.x,  f2 * vrel.y);

    // Build galaxies using your helpers (adds internal rotation etc.)
    create_andromeda_at_position(Vec2(xA, yA), vA, s1, "Primary A");
    create_smaller_galaxy_at_position(Vec2(xB, yB), vB, s2, "Victim B");

    // Make sure nothing bounces off boundaries, and give plenty of room
    PhysicsConfig config;
    config.boundary_min_x = -200.0f;
    config.boundary_max_x = 200.0f;
    config.boundary_min_y = -200.0f;
    config.boundary_max_y = 200.0f;
    config.bounce_force = 0.0f;
    config.time_scale = 0.7f;  // a touch slower = clearer flyby
    if (on_update_physics_params) on_update_physics_params(config);


    std::cout << "✅ Tidal flyby ready: r0=" << r0 << " kpc, rp=" << rp << " kpc\n";
}

// Helper methods for creating galaxies at specific positions
void GalaxyFactory::create_bh_disk_at_position(
    const Vec2& center,
    const Vec2& bulk_vel,
    double bh_mass_1e9Msun,
    int star_count,
    float r_inner_kpc,
    float r_outer_kpc,
    bool clockwise)
{
    // Central SMBH dominates gravity. Give it ALL the mass.
    particle_system_.add_particle(center, bulk_vel,
                                  (float)bh_mass_1e9Msun,
                                  Vec3(1.0f, 1.0f, 0.7f));

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::uniform_real_distribution<float> Th(0.0f, 2.0f * (float)M_PI);

    // Radial distribution ~ r (for a thin disk with more mass in the outskirts)
    auto sample_radius = [&](float r0, float r1) {
        float u = U(rng);
        return std::sqrt(r0*r0 + u * (r1*r1 - r0*r0));
    };

    const float rmin = std::max(r_inner_kpc, 0.05f);  // avoid v→∞
    const float rmax = std::max(r_outer_kpc, rmin + 0.5f);

    // Tiny star mass so self-gravity doesn't unbind the disk
    const float m_star = 1.0e-6f;

    for (int i = 0; i < star_count; ++i) {
        float r  = sample_radius(rmin, rmax);
        float th = Th(rng);

        float x = r * std::cos(th);
        float y = r * std::sin(th);

        // Circular speed from the *central* mass, in your code units
        // v_code = [sqrt(G * M / (r + eps)) in km/s] / VELOCITY_UNIT_KMS
        double v = vcirc_code(r, bh_mass_1e9Msun);  // consistent with your units :contentReference[oaicite:1]{index=1}

        // Tangential direction (flip for clockwise)
        double tx = clockwise ?  std::sin(th) : -std::sin(th);
        double ty = clockwise ? -std::cos(th) :  std::cos(th);

        Vec2 pos = center + Vec2(x, y);
        Vec2 vel = bulk_vel + Vec2((float)(v * tx), (float)(v * ty));

        // Soft pastel disk + golden bulge vibe
        particle_system_.add_particle(pos, vel, m_star, Vec3(0.80f, 0.65f, 1.0f));
    }
}

void GalaxyFactory::create_milky_way_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
    std::normal_distribution<float> velocity_noise(0.0f, 0.3f);
    
    const float disk_scale_length = 3.5f * mass_scale;
    const float bulge_radius = 0.5f * mass_scale;
    
    // Central star
    particle_system_.add_particle(center, velocity, 4.3e6f / 1e9f * mass_scale, Vec3(1.0f, 1.0f, 0.5f));
    
    // Bulge
    int bulge_particles = static_cast<int>(1000 * mass_scale);
    std::gamma_distribution<float> bulge_radius_dist(2.0f, bulge_radius / 2.0f);
    
    for (int i = 0; i < bulge_particles; ++i) {
        float r = std::min(bulge_radius_dist(gen), bulge_radius * 3.0f);
        float theta = angle_dist(gen);
        Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
        
        float v_circ = sqrt(50.0f * mass_scale / (r + 0.1f));
        Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                 v_circ * cos(theta) + velocity_noise(gen));
        
        particle_system_.add_particle(pos, vel, 2.0e10f / bulge_particles / 1e9f * mass_scale, 
                                    Vec3(1.0f, 0.8f, 0.4f));
    }
    
    // Disk
    int disk_particles = static_cast<int>(8000 * mass_scale);
    std::exponential_distribution<float> disk_radius_dist(1.0f / disk_scale_length);
    
    for (int i = 0; i < disk_particles; ++i) {
        float r = std::min(disk_radius_dist(gen), disk_scale_length * 5.0f);
        float theta = angle_dist(gen);
        Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
        
        float v_circ = 22.0f * sqrt(r * mass_scale) / sqrt(r * mass_scale + 1.0f) + 15.0f;
        if (r > 8.0f) v_circ = 22.0f;
        
        Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                 v_circ * cos(theta) + velocity_noise(gen));
        
        particle_system_.add_particle(pos, vel, 5.5e10f / disk_particles / 1e9f * mass_scale,
                                    Vec3(0.8f, 0.6f, 1.0f));
    }
    
    std::cout << "  Created " << name << " at (" << center.x << ", " << center.y << ")\n";
}

void GalaxyFactory::create_andromeda_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
    std::normal_distribution<float> velocity_noise(0.0f, 0.4f);
    
    const float disk_scale_length = 5.0f * mass_scale;
    const float bulge_radius = 1.2f * mass_scale;
    
    // Larger SMBH
    particle_system_.add_particle(center, velocity, 1.4e8f / 1e9f * mass_scale, Vec3(1.0f, 0.9f, 0.3f));
    
    // Larger bulge
    int bulge_particles = static_cast<int>(1500 * mass_scale);
    std::gamma_distribution<float> bulge_radius_dist(2.0f, bulge_radius / 2.0f);
    
    for (int i = 0; i < bulge_particles; ++i) {
        float r = std::min(bulge_radius_dist(gen), bulge_radius * 3.0f);
        float theta = angle_dist(gen);
        Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
        
        float v_circ = sqrt(80.0f * mass_scale / (r + 0.1f));
        Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                 v_circ * cos(theta) + velocity_noise(gen));
        
        particle_system_.add_particle(pos, vel, 4.0e10f / bulge_particles / 1e9f * mass_scale,
                                    Vec3(1.0f, 0.7f, 0.3f));
    }
    
    // Extended disk
    int disk_particles = static_cast<int>(12000 * mass_scale);
    std::exponential_distribution<float> disk_radius_dist(1.0f / disk_scale_length);
    
    for (int i = 0; i < disk_particles; ++i) {
        float r = std::min(disk_radius_dist(gen), disk_scale_length * 6.0f);
        float theta = angle_dist(gen);
        Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
        
        float v_circ = 26.0f * sqrt(r * mass_scale) / sqrt(r * mass_scale + 1.5f) + 20.0f;
        if (r > 10.0f) v_circ = 26.0f;
        
        Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                 v_circ * cos(theta) + velocity_noise(gen));
        
        particle_system_.add_particle(pos, vel, 8.0e10f / disk_particles / 1e9f * mass_scale,
                                    Vec3(0.9f, 0.7f, 1.0f));
    }
    
    std::cout << "  Created " << name << " at (" << center.x << ", " << center.y << ")\n";
}

void GalaxyFactory::create_smaller_galaxy_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
    std::normal_distribution<float> velocity_noise(0.0f, 0.2f);
    
    const float disk_scale_length = 2.0f * mass_scale;
    
    // Small central mass
    particle_system_.add_particle(center, velocity, 1.0e6f / 1e9f * mass_scale, Vec3(0.8f, 0.8f, 1.0f));
    
    // Compact disk
    int disk_particles = static_cast<int>(1500 * mass_scale);
    std::exponential_distribution<float> disk_radius_dist(1.0f / disk_scale_length);
    
    for (int i = 0; i < disk_particles; ++i) {
        float r = std::min(disk_radius_dist(gen), disk_scale_length * 4.0f);
        float theta = angle_dist(gen);
        Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
        
        float v_circ = 15.0f * sqrt(r * mass_scale) / sqrt(r * mass_scale + 0.5f) + 8.0f;
        
        Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                 v_circ * cos(theta) + velocity_noise(gen));
        
        Vec3 color = (name == "LMC") ? Vec3(0.6f, 0.8f, 1.0f) :
                    (name == "SMC") ? Vec3(0.8f, 0.6f, 1.0f) :
                                     Vec3(0.7f, 0.9f, 0.8f);
        
        particle_system_.add_particle(pos, vel, 1.0e10f / disk_particles / 1e9f * mass_scale, color);
    }
    
    std::cout << "  Created " << name << " at (" << center.x << ", " << center.y << ")\n";
}

void GalaxyFactory::create_two_armed_spiral_galaxy(
    const Vec2& center,
    const Vec2& bulk_vel,
    double mass_scale,
    double pitch_deg,
    int disk_particles,
    int bulge_particles,
    int halo_particles,
    double arm_scatter_rad,
    double arm_phase0,
    bool clockwise
) {
    // --- MW-like numbers in 1e9 Msun / kpc, scaled like your MW function ---
    const double scale = 0.01; // same convention you use elsewhere
    const double Rd    = (3.5 / scale) * mass_scale;  // disk scale length (kpc)
    const double a_b   = (0.6 / scale) * mass_scale;  // Hernquist bulge scale (kpc)
    const double rvir  = (200.0 / scale) * mass_scale; // halo "virial" (kpc)
    const double rmax  = (25.0  / scale) * mass_scale; // spawn cutoff (kpc)

    const double M_bh   = (0.0043 / scale) * mass_scale;   // SMBH (1e9 Msun)
    const double M_bul  = (20.0   / scale) * mass_scale;
    const double M_disk = (55.0   / scale) * mass_scale;
    const double M_halo = (1200.0 / scale) * mass_scale;
    const double c_halo = 10.0;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::exponential_distribution<double> Rexp(1.0 / Rd);   // P(r) ~ e^{-r/Rd}
    std::normal_distribution<double>  ThScatter(0.0, arm_scatter_rad);
    std::normal_distribution<double>  Vnoise(0.0, 0.02);    // small vel noise

    auto add_particle = [&](double x, double y, double vx, double vy, double m, const Vec3& col){
        particle_system_.add_particle(center + Vec2{(float)x,(float)y},
                                      bulk_vel + Vec2{(float)vx,(float)vy},
                                      (float)m, col);
    };

    // --- Central SMBH ---
    add_particle(0,0, 0,0, M_bh, Vec3(1.0f,1.0f,0.8f));

    // --- Bulge (Hernquist) ---
    for (int i=0;i<bulge_particles;i++){
        // Inverse CDF for Hernquist: r = a * sqrt(u) / (1 - sqrt(u))
        double u = std::max(1e-12, U01(rng));
        double s = std::sqrt(u);
        double r = std::min(std::max(a_b * s/(1.0 - s), 2.0*DEFAULT_SOFTENING_KPC), rmax);
        double th = 2*M_PI*U01(rng);
        double x = r * std::cos(th), y = r * std::sin(th);

        // Circular speed from total enclosed mass
        double Menc = M_bh
                    + hernquist_Menc(r, M_bul, a_b)
                    + disk_exp_Menc(r, M_disk, Rd)
                    + nfw_Menc(r, M_halo, rvir, c_halo);
        double v = vcirc_code(r, Menc);
        if (clockwise) v = -v;

        add_particle(x,y, -v*std::sin(th)+Vnoise(rng),  v*std::cos(th)+Vnoise(rng),
                     M_bul/double(bulge_particles), Vec3(1.0f,0.8f,0.4f));
    }

    // --- Disk with m=2 logarithmic spiral overdensities ---
    // Log-spiral: r = r0 * exp(b*theta), with b = tan(pitch)
    double pitch = pitch_deg * M_PI/180.0;
    double b = std::tan(pitch);
    double r0 = Rd; // reference radius sets arm phase origin

    for (int i=0;i<disk_particles;i++){
        // Sample radius from exponential disk, clamp
        double r = std::min(std::max(Rexp(rng), 2.0*DEFAULT_SOFTENING_KPC), 4.0*Rd);

        // Choose which arm (0 or 1), arms separated by pi
        int arm = (U01(rng) < 0.5) ? 0 : 1;

        // Ideal arm angle that yields this radius on a log-spiral
        // theta_arm = (ln(r/r0))/b + phase + arm*pi
        double th_arm = (std::log(r / r0))/b + arm_phase0 + arm*M_PI;

        // Scatter around the arm to make a *twin-armed* grand design
        double th = th_arm + ThScatter(rng);
        if (clockwise) th = -th;

        double x = r * std::cos(th), y = r * std::sin(th);

        // Circular speed from enclosed mass (BH + bulge + disk + halo)
        double Menc = M_bh
                    + hernquist_Menc(r, M_bul, a_b)
                    + disk_exp_Menc(r, M_disk, Rd)
                    + nfw_Menc(r, M_halo, rvir, c_halo);
        double v = vcirc_code(r, Menc);
        if (clockwise) v = -v;

        double vx = -v*std::sin(th) + Vnoise(rng);
        double vy =  v*std::cos(th) + Vnoise(rng);

        add_particle(x,y, vx,vy, M_disk/double(disk_particles), Vec3(0.8f,0.65f,1.0f));
    }

    // --- Simple quasi-isotropic halo sampling (projected to 2D sim plane) ---
    for (int i=0;i<halo_particles;i++){
        // Sample radius with a crude rejection to roughly follow NFW M(r)
        double r, accepted = false;
        const double rs = rvir / c_halo;
        for (int tries=0; tries<40 && !accepted; ++tries){
            r = U01(rng) * rmax;
            double x = r/rs;
            double f = std::log(1.0+x) - x/(1.0+x);
            double fmax = std::log(1.0 + rmax/rs);
            accepted = (U01(rng) <= std::max(0.0, f)/std::max(1e-6, fmax));
        }
        r = std::max(r, 2.0*DEFAULT_SOFTENING_KPC);
        double th = 2*M_PI*U01(rng);
        if (clockwise) th = -th;

        // simple, small velocity dispersion that falls with r
        double sigma_code = (100.0 / VELOCITY_UNIT_KMS) / std::sqrt(1.0 + r/rs);
        std::normal_distribution<double> Vd(0.0, sigma_code);

        add_particle(r*std::cos(th), r*std::sin(th), Vd(rng), Vd(rng),
                     M_halo/double(halo_particles), Vec3(0.25f,0.12f,0.5f));
    }
}

// Classic preset configurations
void GalaxyFactory::create_galaxy_spiral() {
    particle_system_.clear_particles();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.5f);
    
    for (int i = 0; i < 5000; ++i) {
        float angle = (i / 5000.0f) * 8 * M_PI;
        float radius = (i / 5000.0f) * 15.0f + 2.0f;
        
        float x = radius * cos(angle) + noise(gen);
        float y = radius * sin(angle) + noise(gen);
        
        // Orbital velocity
        float vel_magnitude = sqrt(10.0f / radius);  // Simplified orbital mechanics
        float vel_x = -vel_magnitude * sin(angle) + noise(gen) * 0.1f;
        float vel_y = vel_magnitude * cos(angle) + noise(gen) * 0.1f;
        
        Vec2 pos(x, y);
        Vec2 vel(vel_x, vel_y);
        Vec3 color(0.8f + noise(gen) * 0.2f, 0.6f + noise(gen) * 0.3f, 1.0f);
        
        particle_system_.add_particle(pos, vel, 1.0f, color);
    }
    
    // Add central massive object
    particle_system_.add_particle(Vec2(0, 0), Vec2(0, 0), 100.0f, Vec3(1.0f, 1.0f, 0.5f));
    
    
    configure_for_galaxy_scale();
}

void GalaxyFactory::create_solar_system() {
    particle_system_.clear_particles();
    
    // Central star
    particle_system_.add_particle(Vec2(0, 0), Vec2(0, 0), 50.0f, Vec3(1.0f, 1.0f, 0.3f));
    
    // Planets
    std::vector<float> planet_distances = {3, 5, 8, 12, 16};
    std::vector<Vec3> planet_colors = {
        Vec3(0.8f, 0.4f, 0.2f),  // Mercury-like
        Vec3(0.9f, 0.7f, 0.3f),  // Venus-like
        Vec3(0.3f, 0.5f, 1.0f),  // Earth-like
        Vec3(0.8f, 0.2f, 0.1f),  // Mars-like
        Vec3(0.9f, 0.8f, 0.6f)   // Jupiter-like
    };
    
    for (size_t i = 0; i < planet_distances.size(); ++i) {
        float distance = planet_distances[i];
        float velocity = sqrt(20.0f / distance);  // Orbital velocity
        
        particle_system_.add_particle(
            Vec2(distance, 0), 
            Vec2(0, velocity), 
            2.0f, 
            planet_colors[i]
        );
    }
    
    configure_for_solar_system_scale();
}

void GalaxyFactory::create_cluster() {
    particle_system_.clear_particles();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> pos_dist(0.0f, 5.0f);
    std::normal_distribution<float> vel_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> color_dist(0.5f, 1.0f);
    
    for (int i = 0; i < 3000; ++i) {
        Vec2 pos(pos_dist(gen), pos_dist(gen));
        Vec2 vel(vel_dist(gen), vel_dist(gen));
        Vec3 color(color_dist(gen), color_dist(gen), color_dist(gen));
        
        particle_system_.add_particle(pos, vel, 1.0f, color);
    }
    
    
    configure_for_galaxy_scale();
}

void GalaxyFactory::add_random_particles(int count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-20.0f, 20.0f);
    std::uniform_real_distribution<float> vel_dist(-3.0f, 3.0f);
    std::uniform_real_distribution<float> color_dist(0.2f, 1.0f);
    std::uniform_real_distribution<float> mass_dist(0.8f, 1.2f);
    
    for (int i = 0; i < count; ++i) {
        Vec2 pos(pos_dist(gen), pos_dist(gen));
        Vec2 vel(vel_dist(gen), vel_dist(gen));
        Vec3 color(color_dist(gen), color_dist(gen), color_dist(gen));
        float mass = mass_dist(gen);
        
        if (!particle_system_.add_particle(pos, vel, mass, color)) {
            std::cout << "Reached maximum particle limit (" << particle_system_.get_max_particles() << ")\n";
            break;
        }
    }
}

// Physics and camera configuration methods
void GalaxyFactory::configure_for_galaxy_scale() {
    PhysicsConfig config;
    config.boundary_min_x = -50.0f;
    config.boundary_max_x = 50.0f;
    config.boundary_min_y = -50.0f;
    config.boundary_max_y = 50.0f;
    config.bounce_force = 0.0f;
    config.damping = 1.0f;
    config.time_scale = 0.5f;
    
    if (on_update_physics_params) {
        on_update_physics_params(config);
    }
    
    if (on_update_camera) {
        on_update_camera(0.0, 0.0, 15.0); // Zoom out to see whole galaxy
    }
}

void GalaxyFactory::configure_for_local_group_scale() {
    PhysicsConfig config;
    config.boundary_min_x = -100.0f;
    config.boundary_max_x = 100.0f;
    config.boundary_min_y = -100.0f;
    config.boundary_max_y = 100.0f;
    config.bounce_force = 0.0f;
    config.damping = 1.0f;
    config.time_scale = 0.3f;
    
    if (on_update_physics_params) {
        on_update_physics_params(config);
    }
    
    if (on_update_camera) {
        on_update_camera(0.0, 0.0, 5.0); // Zoom out to see multiple galaxies
    }
}

void GalaxyFactory::configure_for_solar_system_scale() {
    PhysicsConfig config;
    config.boundary_min_x = -25.0f;
    config.boundary_max_x = 25.0f;
    config.boundary_min_y = -25.0f;
    config.boundary_max_y = 25.0f;
    config.bounce_force = 1000.0f;
    config.damping = 1.0f;
    config.time_scale = 1.0f;
    
    if (on_update_physics_params) {
        on_update_physics_params(config);
    }
    
    if (on_update_camera) {
        on_update_camera(0.0, 0.0, 25.0); // Zoom in to see solar system
    }
}
