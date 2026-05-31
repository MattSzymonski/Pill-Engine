// PBR static vertex shader. Edit here — pill_assets regenerates the .wgsl.
// Per-draw storage holds raw position/rotation(radians)/scale; the GPU builds the
// model matrix here (hardware sin/cos), so the CPU does ZERO per-entity trig and
// uploads 48 B instead of a 64 B matrix. MVP = viewProjection·model, also on GPU.
// [Aaltonen "HypeHype" GDC 2023; Lottes @NOTimothyLottes 2025-01-23]

struct Camera {
    float4                position;
    column_major float4x4 viewProjection;
    float3                fog_color;
    float                 fog_density;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> UCamera;

struct PerDraw {
    float4 position; // xyz
    float4 rotation; // xyz, radians
    float4 scale;    // xyz
};
[[vk::binding(0, 3)]] StructuredBuffer<PerDraw> UPerDrawArray;

struct VSOut {
    [[vk::location(0)]] float2 uv         : TEXCOORD0;
    [[vk::location(1)]] float3 worldPos   : TEXCOORD1;
    [[vk::location(2)]] float3 worldNormal: TEXCOORD2;
                        float4 sv_position: SV_POSITION;
};

struct VSIn {
    [[vk::location(0)]] float3 pos;
    [[vk::location(4)]] float2 uv;
    [[vk::location(5)]] float3 normal;
};

// Row-major float3x3 constructors; mul(R, v) applies standard right-handed rotation.
float3x3 rot_x(float a) { float c = cos(a), s = sin(a); return float3x3(1, 0, 0,  0, c, -s,  0, s, c); }
float3x3 rot_y(float a) { float c = cos(a), s = sin(a); return float3x3(c, 0, s,  0, 1, 0,  -s, 0, c); }
float3x3 rot_z(float a) { float c = cos(a), s = sin(a); return float3x3(c, -s, 0,  s, c, 0,  0, 0, 1); }

[shader("vertex")]
VSOut vs_main(VSIn input, uint instance_id : SV_InstanceID) {
    PerDraw per_draw = UPerDrawArray[instance_id];

    // model = T * (Rx*Ry*Rz) * S — matches glam from_scale_rotation_translation(Quat::x*y*z).
    float3x3 rotation = mul(rot_x(per_draw.rotation.x), mul(rot_y(per_draw.rotation.y), rot_z(per_draw.rotation.z)));
    float3 scaled = input.pos * per_draw.scale.xyz;
    float3 worldPos = per_draw.position.xyz + mul(rotation, scaled);
    float4 worldPos4 = float4(worldPos, 1.0);

    VSOut output;
    output.sv_position = mul(UCamera.viewProjection, worldPos4);
    output.uv          = input.uv;
    output.worldPos    = worldPos;
    output.worldNormal = normalize(mul(rotation, input.normal));
    return output;
}
