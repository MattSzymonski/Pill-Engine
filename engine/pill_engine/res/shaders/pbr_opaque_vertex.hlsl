// PBR vertex shader (pill_assets regenerates the .wgsl). The model matrix is built here from
// per-draw position / rotation (radians) / scale, so the CPU uploads raw transforms.

struct Camera {
    float4                position;
    column_major float4x4 view_projection;
    column_major float4x4 previous_view_projection;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;

struct DrawTransform {
    float4 position; // xyz
    float4 rotation; // xyz, radians
    float4 scale;    // xyz
};
// This and last frame's transform (motion vectors); mirrors PerDrawStd140 in pass_pbr_opaque.rs
struct PerDraw {
    DrawTransform current;
    DrawTransform previous;
};
[[vk::binding(0, 3)]] StructuredBuffer<PerDraw> per_draw_array;

struct VertexOutput {
    [[vk::location(0)]] float2 texture_coordinates : TEXCOORD0;
    [[vk::location(1)]] float3 world_position      : TEXCOORD1;
    [[vk::location(2)]] float3 world_normal        : TEXCOORD2;
    [[vk::location(3)]] float3 world_tangent       : TEXCOORD3;
    [[vk::location(4)]] float3 world_bitangent     : TEXCOORD4;
    [[vk::location(5)]] float2 velocity            : TEXCOORD5;
                        float4 sv_position         : SV_POSITION;
};

struct VertexInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(4)]] float2 texture_coordinates;
    [[vk::location(5)]] float3 normal;
    [[vk::location(6)]] float3 tangent;
    [[vk::location(7)]] float3 bitangent;
};

// Row-major float3x3 constructors; mul(R, v) applies standard right-handed rotation.
float3x3 rot_x(float angle) { float cosine = cos(angle), sine = sin(angle); return float3x3(1, 0, 0,  0, cosine, -sine,  0, sine, cosine); }
float3x3 rot_y(float angle) { float cosine = cos(angle), sine = sin(angle); return float3x3(cosine, 0, sine,  0, 1, 0,  -sine, 0, cosine); }
float3x3 rot_z(float angle) { float cosine = cos(angle), sine = sin(angle); return float3x3(cosine, -sine, 0,  sine, cosine, 0,  0, 0, 1); }

// Zero for (near) zero-length input, so degenerate tangents don't turn into NaN
float3 safe_normalize(float3 value) {
    float length_squared = dot(value, value);
    return length_squared > 1e-12 ? value * rsqrt(length_squared) : float3(0.0, 0.0, 0.0);
}

// model = T * (Rx*Ry*Rz) * S — matches glam from_scale_rotation_translation(Quat::x*y*z).
float3x3 rotation_matrix(DrawTransform transform) {
    return mul(rot_x(transform.rotation.x), mul(rot_y(transform.rotation.y), rot_z(transform.rotation.z)));
}

[shader("vertex")]
VertexOutput vs_main(VertexInput input, uint instance_id : SV_InstanceID) {
    PerDraw per_draw = per_draw_array[instance_id];
    DrawTransform transform = per_draw.current;

    float3x3 rotation = rotation_matrix(transform);
    float3 world_position = transform.position.xyz + mul(rotation, input.position * transform.scale.xyz);
    float4 clip_position  = mul(camera.view_projection, float4(world_position, 1.0));

    DrawTransform previous = per_draw.previous;
    float3 previous_world_position = previous.position.xyz + mul(rotation_matrix(previous), input.position * previous.scale.xyz);
    float4 previous_clip_position  = mul(camera.previous_view_projection, float4(previous_world_position, 1.0));

    VertexOutput output;
    output.sv_position         = clip_position;
    // Motion vector in UV units (NDC y points up, UV v points down); interpolated per pixel
    output.velocity            = (clip_position.xy / clip_position.w - previous_clip_position.xy / previous_clip_position.w) * float2(0.5, -0.5);
    output.texture_coordinates = input.texture_coordinates;
    output.world_position      = world_position;
    // Inverse-transpose of R*S is R*S^-1: divide by scale so non-uniform scale keeps normals perpendicular
    output.world_normal        = normalize(mul(rotation, input.normal / transform.scale.xyz));
    // Tangents lie in the surface, so they transform like positions (R*S)
    output.world_tangent       = safe_normalize(mul(rotation, input.tangent * transform.scale.xyz));
    output.world_bitangent     = safe_normalize(mul(rotation, input.bitangent * transform.scale.xyz));
    return output;
}
