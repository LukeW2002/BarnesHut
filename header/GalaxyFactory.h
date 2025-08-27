#pragma once

#include "Vec2.h"
#include "BarnesHutParticleSystem.h"

#include <string>
#include <functional>



// Forward declarations
//class BarnesHutParticleSystem;

// Physical constants for galaxy simulations
//namespace GalaxyConstants {
//    // World-scale constants for galaxy simulations
//    static constexpr double MILKY_WAY_RADIUS_KPC = 25.0;      // 25 kpc galactic radius
//    static constexpr double GALACTIC_BOUNDARY_KPC = 40.0;     // 40 kpc simulation boundary
//    static constexpr double SOLAR_CIRCLE_RADIUS_KPC = 8.2;    // Sun's distance from galactic center
//    static constexpr double FLAT_ROTATION_VELOCITY_KMS = 220.0; // Flat rotation curve velocity
//    
//    // Physical unit conversions (from your codebase)
//    static constexpr double G_GALACTIC = 4.3009;              // kpc·(km/s)²·(10⁹M☉)⁻¹  
//    static constexpr double VELOCITY_UNIT_KMS = 65.58;        // km/s per code velocity unit
//    static constexpr double TIME_UNIT_MYR = 14.91;            // Myr per code time unit
//    static constexpr double DEFAULT_SOFTENING_KPC = 0.050;    // 50 pc = 0.05 kpc softening
//}

// Factory class for creating realistic galaxy simulations
class GalaxyFactory {
public:
    // Constructor takes reference to particle system
    explicit GalaxyFactory(BarnesHutParticleSystem& particle_system);
    
    // Individual galaxy creation methods
    void create_realistic_milky_way();
    void create_andromeda_like_galaxy();
    
    // Galaxy interaction scenarios
    void create_galaxy_merger_scenario();
    void create_local_group_simulation();
    void create_twin_spiral_pair();
    void create_dwarf_galaxy_accretion();
    void create_tidally_disrupting_flyby();
    void create_two_spiral_binary_merger_100k();
    
    // Classic preset configurations
    void create_galaxy_spiral();
    void create_solar_system();
    void create_cluster();
    void add_random_particles(int count);
    
    // Physics parameter configuration
    struct PhysicsConfig {
        float boundary_min_x = -50.0f;
        float boundary_max_x = 50.0f;
        float boundary_min_y = -50.0f;
        float boundary_max_y = 50.0f;
        float bounce_force = 0.0f;
        float damping = 1.0f;
        float time_scale = 0.5f;
    };
    
    // Callback to update physics parameters (set by main app)
    std::function<void(const PhysicsConfig&)> on_update_physics_params;
    
    // Camera configuration callback
    std::function<void(double center_x, double center_y, double zoom)> on_update_camera;

private:
    BarnesHutParticleSystem& particle_system_;
    
    // Helper methods for mass profile calculations
    static double hernquist_Menc(double r, double Mb, double a);
    static double disk_exp_Menc(double r, double Md, double Rd);
    static double nfw_Menc(double r, double Mh, double rvir, double c);
    static double vcirc_code(double r_kpc, double Menc_1e9Msun);
    static double clamp_min_r(double r, double rmin);
    
    // Helper methods for creating galaxies at specific positions
    void create_milky_way_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name);
    void create_andromeda_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name);
    void create_smaller_galaxy_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name);
    void create_bh_disk_at_position(
        const Vec2& center,
        const Vec2& bulk_vel,
        double bh_mass_1e9Msun,   // central mass (in 1e9 Msun)
        int star_count,            // number of disk stars
        float r_inner_kpc,         // inner cutoff (avoid crazy speeds)
        float r_outer_kpc,         // outer edge of disk
        bool clockwise);
    void create_bh_ring_galaxy(
        const Vec2& center,
        const Vec2& bulk_vel,
        double bh_mass_1e9Msun, // mass of central body in 1e9 Msun
        int    star_count,      // number of light particles
        float  r_inner_kpc,     // inner cutoff (avoid crazy speeds)
        float  r_outer_kpc,     // outer radius of disk
        float  core_eps_kpc,
        bool   clockwise);      // spin direction
    
    // Advanced spiral galaxy generator
    void create_two_armed_spiral_galaxy(
        const Vec2& center,
        const Vec2& bulk_vel,
        double mass_scale = 1.0,
        double pitch_deg = 16.0,
        int disk_particles = 60000,
        int bulge_particles = 3000,
        int halo_particles = 25000,
        double arm_scatter_rad = 0.18,
        double arm_phase0 = 0.0,
        bool clockwise = false
    );
    
    // Update physics and camera with recommended settings
    void configure_for_galaxy_scale();
    void configure_for_local_group_scale();
    void configure_for_solar_system_scale();
};
