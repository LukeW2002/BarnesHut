#pragma once
#include "Vec2.h"
#include "EventSystem.h"
#include <vector>
#include <memory>

class ParticleSystem {
public:
    ParticleSystem(size_t max_particles, EventBus& event_bus);
    
    // Particle management
    bool add_particle(const Vec2& pos, const Vec2& vel, float mass, const Vec3& color);
    void clear_particles();
    
    // Physics simulation
    void update(float dt);
    void set_boundary(float min_x, float max_x, float min_y, float max_y);
    
    // Configuration
    void set_bounce_force(float force) { bounce_force_ = force; }
    void set_damping(float damping) { damping_ = damping; }
    void set_gravity(const Vec2& gravity) { gravity_ = gravity; }
    
    // Access for testing and debugging
    size_t get_particle_count() const { return particle_count_; }
    size_t get_max_particles() const { return max_particles_; }
    
    const std::vector<float>& get_positions_x() const { return positions_x_; }
    const std::vector<float>& get_positions_y() const { return positions_y_; }
    const std::vector<float>& get_velocities_x() const { return velocities_x_; }
    const std::vector<float>& get_velocities_y() const { return velocities_y_; }
    const std::vector<float>& get_forces_x() const { return forces_x_; }
    const std::vector<float>& get_forces_y() const { return forces_y_; }
    const std::vector<float>& get_masses() const { return masses_; }
    
    // Get individual particle data
    Vec2 get_position(size_t index) const;
    Vec2 get_velocity(size_t index) const;
    Vec2 get_force(size_t index) const;
    float get_mass(size_t index) const;
    Vec3 get_color(size_t index) const;
    
private:
    void apply_boundary_forces();
    void apply_gravity_forces();
    void integrate_leapfrog(float dt);
    void prepare_render_data();
    void calculate_bounds();
    
    size_t max_particles_;
    size_t particle_count_;
    EventBus& event_bus_;
    
    // Simulation parameters
    float bounce_force_;
    float damping_;
    Vec2 gravity_;
    float bounds_min_x_, bounds_max_x_;
    float bounds_min_y_, bounds_max_y_;
    
    // SOA storage for cache efficiency (physics hot path)
    std::vector<float> positions_x_, positions_y_;
    std::vector<float> velocities_x_, velocities_y_;
    std::vector<float> forces_x_, forces_y_;
    std::vector<float> masses_;
    std::vector<float> colors_r_, colors_g_, colors_b_;
    
    // Render data (interleaved for GPU)
    std::vector<float> render_positions_;  // [x0,y0,x1,y1,...]
    std::vector<float> render_colors_;     // [r0,g0,b0,r1,g1,b1,...]
    
    // Performance tracking
    size_t iteration_count_;
};

