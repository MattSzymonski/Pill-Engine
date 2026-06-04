// Equirectangular UV mapping: atan2(z,x) convention matches the bake in studio_equirect.rs.
static const float EQUIRECT_PI = 3.14159265359;

float2 dir_to_equirect_uv(float3 dir) {
    float3 d = normalize(dir);
    float  u = 0.5 + atan2(d.z, d.x) / (2.0 * EQUIRECT_PI);
    float  v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / EQUIRECT_PI;
    return float2(frac(u), clamp(v, 0.0, 1.0));
}
