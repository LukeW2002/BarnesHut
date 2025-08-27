#pragma once
#include <cmath>

// TODO:  be replaced by Eigen 
struct Vec2 {
    float x, y;
    
    Vec2() : x(0), y(0) {}
    Vec2(float x, float y) : x(x), y(y) {}
    
    Vec2 operator+(const Vec2& other) const { return Vec2(x + other.x, y + other.y); }
    Vec2 operator-(const Vec2& other) const { return Vec2(x - other.x, y - other.y); }
    Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
    Vec2 operator/(float scalar) const { return Vec2(x / scalar, y / scalar); }
    
    Vec2& operator+=(const Vec2& other) { x += other.x; y += other.y; return *this; }
    Vec2& operator-=(const Vec2& other) { x -= other.x; y -= other.y; return *this; }
    Vec2& operator*=(float scalar) { x *= scalar; y *= scalar; return *this; }
    
    float length() const { return std::sqrt(x*x + y*y); }
    float length_squared() const { return x*x + y*y; }
    Vec2 normalized() const { 
        float len = length(); 
        return len > 0 ? *this / len : Vec2(); 
    }
    
    float dot(const Vec2& other) const { return x * other.x + y * other.y; }
    float cross(const Vec2& other) const { return x * other.y - y * other.x; }
    
    // Useful for boundary checks
    bool is_within_bounds(float min_x, float max_x, float min_y, float max_y) const {
        return x >= min_x && x <= max_x && y >= min_y && y <= max_y;
    }
};

// Color as 3-component vector (RGB)
struct Vec3 {
    float x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator-(const Vec3& other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
    Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    
    Vec3& operator+=(const Vec3& other) { x += other.x; y += other.y; z += other.z; return *this; }
    Vec3& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
    
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3 normalized() const { 
        float len = length(); 
        return len > 0 ? *this * (1.0f/len) : Vec3(); 
    }
};
