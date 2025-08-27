#pragma once
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>

// Event system for decoupled communication
class EventBus {
public:
    using EventHandler = std::function<void(const void* data)>;
    
    template<typename T>
    void subscribe(const std::string& event_type, std::function<void(const T&)> handler) {
        handlers_[event_type].push_back([handler](const void* data) {
            handler(*static_cast<const T*>(data));
        });
    }
    
    template<typename T>
    void emit(const std::string& event_type, const T& data) {
        auto it = handlers_.find(event_type);
        if (it != handlers_.end()) {
            for (auto& handler : it->second) {
                handler(&data);
            }
        }
    }
    
    // For testing - check if event type has subscribers
    bool has_subscribers(const std::string& event_type) const {
        auto it = handlers_.find(event_type);
        return it != handlers_.end() && !it->second.empty();
    }
    
    size_t get_subscriber_count(const std::string& event_type) const {
        auto it = handlers_.find(event_type);
        return it != handlers_.end() ? it->second.size() : 0;
    }
    
private:
    std::unordered_map<std::string, std::vector<EventHandler>> handlers_;
};

// Physics events
struct PhysicsUpdateEvent {
    float delta_time;
    size_t particle_count;
    size_t iteration_count;
};

struct RenderUpdateEvent {
    const float* positions;  // x,y interleaved: [x0,y0,x1,y1,...]
    const float* colors;     // r,g,b interleaved: [r0,g0,b0,r1,g1,b1,...]
    size_t particle_count;
    float bounds_min_x, bounds_max_x;
    float bounds_min_y, bounds_max_y;
};

struct ParticleAddedEvent {
    size_t particle_index;
    float pos_x, pos_y;
    float vel_x, vel_y;
    float mass;
    float color_r, color_g, color_b;
};

// Event type constants to avoid string typos
namespace Events {
    constexpr const char* PHYSICS_UPDATE = "physics_update";
    constexpr const char* RENDER_UPDATE = "render_update";
    constexpr const char* PARTICLE_ADDED = "particle_added";
    constexpr const char* SIMULATION_PAUSED = "simulation_paused";
    constexpr const char* SIMULATION_RESET = "simulation_reset";
}
