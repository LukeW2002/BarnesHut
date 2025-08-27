// Implementation
#include "ParticleSystem.h"
#include <algorithm>
#include <cmath>

ParticleSystem::ParticleSystem(size_t max_particles, EventBus& event_bus) 
    : max_particles_(max_particles), 
      particle_count_(0), 
      event_bus_(event_bus),
      bounce_force_(1000.0f),
      damping_(0.999f),
      gravity_(0.0f, 0.0f),
      bounds_min_x_(-10.0f), bounds_max_x_(10.0f),
      bounds_min_y_(-10.0f), bounds_max_y_(10.0f),
      iteration_count_(0) {
    
    // Allocate SOA arrays
    positions_x_.resize(max_particles);
    positions_y_.resize(max_particles);
    velocities_x_.resize(max_particles);
    velocities_y_.resize(max_particles);
    forces_x_.resize(max_particles);
    forces_y_.resize(max_particles);
    masses_.resize(max_particles);
    colors_r_.resize(max_particles);
    colors_g_.resize(max_particles);
    colors_b_.resize(max_particles);
    
    // For rendering - interleaved data
    render_positions_.resize(max_particles * 2);
    render_colors_.resize(max_particles * 3);
}

bool ParticleSystem::add_particle(const Vec2& pos, const Vec2& vel, float mass, const Vec3& color) {
    if (particle_count_ >= max_particles_) {
        return false;
    }
    
    size_t idx = particle_count_;
    positions_x_[idx] = pos.x;
    positions_y_[idx] = pos.y;
    velocities_x_[idx] = vel.x;
    velocities_y_[idx] = vel.y;
    forces_x_[idx] = 0.0f;
    forces_y_[idx] = 0.0f;
    masses_[idx] = mass;
    colors_r_[idx] = color.x;
    colors_g_[idx] = color.y;
    colors_b_[idx] = color.z;
    
    particle_count_++;
    
    // Emit particle added event
    ParticleAddedEvent event{idx, pos.x, pos.y, vel.x, vel.y, mass, color.x, color.y, color.z};
    event_bus_.emit(Events::PARTICLE_ADDED, event);
    
    return true;
}

void ParticleSystem::clear_particles() {
    particle_count_ = 0;
    iteration_count_ = 0;
}

void ParticleSystem::set_boundary(float min_x, float max_x, float min_y, float max_y) {
    bounds_min_x_ = min_x;
    bounds_max_x_ = max_x;
    bounds_min_y_ = min_y;
    bounds_max_y_ = max_y;
}

void ParticleSystem::update(float dt) {
    if (particle_count_ == 0) return;
    
    // Clear forces from previous frame
    std::fill(forces_x_.begin(), forces_x_.begin() + particle_count_, 0.0f);
    std::fill(forces_y_.begin(), forces_y_.begin() + particle_count_, 0.0f);
    
    // Apply forces (hot path - direct calls for performance)
    apply_boundary_forces();
    apply_gravity_forces();
    
    // Integrate physics using Leapfrog
    integrate_leapfrog(dt);
    
    // Prepare data for rendering
    prepare_render_data();
    
    iteration_count_++;
    
    // Emit events (not in hot path)
    PhysicsUpdateEvent physics_event{dt, particle_count_, iteration_count_};
    event_bus_.emit(Events::PHYSICS_UPDATE, physics_event);
    
    calculate_bounds();
    RenderUpdateEvent render_event{
        render_positions_.data(), 
        render_colors_.data(), 
        particle_count_,
        bounds_min_x_, bounds_max_x_,
        bounds_min_y_, bounds_max_y_
    };
    event_bus_.emit(Events::RENDER_UPDATE, render_event);
}

void ParticleSystem::apply_boundary_forces() {
    for (size_t i = 0; i < particle_count_; ++i) {
        // X boundaries
        if (positions_x_[i] > bounds_max_x_) {
            forces_x_[i] -= bounce_force_ * (positions_x_[i] - bounds_max_x_);
        } else if (positions_x_[i] < bounds_min_x_) {
            forces_x_[i] -= bounce_force_ * (positions_x_[i] - bounds_min_x_);
        }
        
        // Y boundaries
        if (positions_y_[i] > bounds_max_y_) {
            forces_y_[i] -= bounce_force_ * (positions_y_[i] - bounds_max_y_);
        } else if (positions_y_[i] < bounds_min_y_) {
            forces_y_[i] -= bounce_force_ * (positions_y_[i] - bounds_min_y_);
        }
    }
}

void ParticleSystem::apply_gravity_forces() {
    if (gravity_.length_squared() > 0) {
        for (size_t i = 0; i < particle_count_; ++i) {
            forces_x_[i] += gravity_.x * masses_[i];
            forces_y_[i] += gravity_.y * masses_[i];
        }
    }
}

void ParticleSystem::integrate_leapfrog(float dt) {
    for (size_t i = 0; i < particle_count_; ++i) {
        float inv_mass = 1.0f / masses_[i];
        
        // Kick (update velocities)
        velocities_x_[i] += forces_x_[i] * inv_mass * dt;
        velocities_y_[i] += forces_y_[i] * inv_mass * dt;
        
        // Apply damping
        velocities_x_[i] *= damping_;
        velocities_y_[i] *= damping_;
        
        // Drift (update positions)
        positions_x_[i] += velocities_x_[i] * dt;
        positions_y_[i] += velocities_y_[i] * dt;
    }
}

void ParticleSystem::prepare_render_data() {
    for (size_t i = 0; i < particle_count_; ++i) {
        render_positions_[i * 2 + 0] = positions_x_[i];
        render_positions_[i * 2 + 1] = positions_y_[i];
        
        render_colors_[i * 3 + 0] = colors_r_[i];
        render_colors_[i * 3 + 1] = colors_g_[i];
        render_colors_[i * 3 + 2] = colors_b_[i];
    }
}

void ParticleSystem::calculate_bounds() {
    if (particle_count_ == 0) return;
    
    float min_x = positions_x_[0], max_x = positions_x_[0];
    float min_y = positions_y_[0], max_y = positions_y_[0];
    
    for (size_t i = 1; i < particle_count_; ++i) {
        min_x = std::min(min_x, positions_x_[i]);
        max_x = std::max(max_x, positions_x_[i]);
        min_y = std::min(min_y, positions_y_[i]);
        max_y = std::max(max_y, positions_y_[i]);
    }
    
    bounds_min_x_ = min_x;
    bounds_max_x_ = max_x;
    bounds_min_y_ = min_y;
    bounds_max_y_ = max_y;
}

// Accessor methods
Vec2 ParticleSystem::get_position(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(positions_x_[index], positions_y_[index]);
}

Vec2 ParticleSystem::get_velocity(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(velocities_x_[index], velocities_y_[index]);
}

Vec2 ParticleSystem::get_force(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(forces_x_[index], forces_y_[index]);
}

float ParticleSystem::get_mass(size_t index) const {
    if (index >= particle_count_) return 0.0f;
    return masses_[index];
}

Vec3 ParticleSystem::get_color(size_t index) const {
    if (index >= particle_count_) return Vec3();
    return Vec3(colors_r_[index], colors_g_[index], colors_b_[index]);
}
