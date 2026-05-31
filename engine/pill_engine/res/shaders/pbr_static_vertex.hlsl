// PBR static vertex shader. Edit here — pill_assets regenerates the .wgsl.
// Reads pos/uv/normal from vertex buffer (locations 0, 4, 5); MVP and model
// matrix come from the per-draw storage array at group 3 binding 0,
// indexed by SV_InstanceID so one instanced draw covers a full material batch.

struct Camera {
    float4                position;
    column_major float4x4 viewProjection;
    float3                fog_color;
    float                 fog_density;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> UCamera;

struct PerDraw {
    column_major float4x4 mvp;
    column_major float4x4 model;
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

[shader("vertex")]
VSOut vs_main(VSIn input, uint instance_id : SV_InstanceID) {
    PerDraw per_draw = UPerDrawArray[instance_id];
    float4 worldPos4 = mul(per_draw.model, float4(input.pos, 1.0));
    // TODO: Use a proper normal matrix (inverse-transpose of model) if non-uniform scaling is used.
    float3 n = normalize(mul(per_draw.model, float4(input.normal, 0.0)).xyz);
    VSOut output;
    output.sv_position = mul(per_draw.mvp, float4(input.pos, 1.0));
    output.uv          = input.uv;
    output.worldPos    = worldPos4.xyz;
    output.worldNormal = n;
    return output;
}
