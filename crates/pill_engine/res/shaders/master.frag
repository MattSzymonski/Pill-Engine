#version 450

// Input vertex data
layout(location=0) in vec3 vertex_position;
layout(location=1) in vec2 vertex_texture_coordinates;
layout(location=2) in mat3 TBN_matrix;

//layout(location=2) in vec3 vertex_light_position; // NEW!
//layout(location=3) in vec3 vertex_view_position; // NEW!



// Input vertex data
// layout(location=0) in vec3 vertex_position;
// layout(location=1) in vec2 vertex_texture_coordinates;
// layout(location=2) in vec3 vertex_normal;
// layout(location=3) in vec3 vertex_tangent;
// layout(location=4) in vec3 vertex_bitangent;

// Input material data
layout(set = 0, binding = 0) uniform texture2D diffuse_texture;
layout(set = 0, binding = 1) uniform sampler diffuse_sampler;
layout(set = 0, binding = 2) uniform texture2D normal_texture;
layout(set = 0, binding = 3) uniform sampler normal_sampler;
layout(set = 1, binding = 0) uniform properties {
    vec3 tint;
};

// Output data
layout(location=0) out vec4 out_final_color;


// layout(set = 2, binding = 0) uniform Light {
//     vec3 light_position;
//     vec3 light_color;
// };


// Input camera data
layout(set=2, binding=0) uniform camera {
    vec3 camera_position; 
    mat4 camera_view_projection;
};

void main() {

    // Settings
    float ambient_light_strength = 0.1;
    vec3 light_position = vec3(10.0, 10.0, 10.0);
    vec3 light_color = vec3(1.0, 1.0, 1.0);


    // Texture
    vec4 object_color = texture(sampler2D(diffuse_texture, diffuse_sampler), vertex_texture_coordinates);
    vec4 object_normal = texture(sampler2D(normal_texture, normal_sampler), vertex_texture_coordinates);

    // Ambient lighting
    vec3 ambient_light_factor = light_color * ambient_light_strength;

    // Diffuse lighting
    vec3 normal = normalize(object_normal.rgb * 2.0 - 1.0); // Transform normal vector to (-1,1) range 
    vec3 light_direction = normalize(TBN_matrix * light_position - vertex_position);

    float diffuse_light_strength = max(dot(normal, light_direction), 0.0);
    vec3 diffuse_light_factor = light_color * diffuse_light_strength;

    // Specular lighting
    vec3 view_direction = normalize(TBN_matrix * camera_position - vertex_position);
    //vec3 half_direction = normalize(view_direction + light_direction);
    vec3 reflection_direction = reflect(-light_direction, normal);

    float specular_light_strength = pow(max(dot(view_direction, reflection_direction), 0.0), 32);
    //float specular_light_strength = pow(max(dot(normal, half_direction), 0.0), 32);
    vec3 specular_light_factor = light_color * specular_light_strength * 2.0;

    
    // Final color
    vec3 final_color = (ambient_light_factor + diffuse_light_factor + specular_light_factor) * object_color.xyz * tint;
    out_final_color = vec4(final_color, 1.0);

    // vec4 object_normal = texture(sampler2D(t_normal, s_normal), v_tex_coords);

    // float ambient_strength = 0.1;
    // vec3 ambient_color = light_color * ambient_strength;

    // vec3 normal = normalize(object_normal.rgb * 2.0 - 1.0); // UPDATED!
    // vec3 light_dir = normalize(v_light_position - v_position); // UPDATED!
    
    // float diffuse_strength = max(dot(normal, light_dir), 0.0);
    // vec3 diffuse_color = light_color * diffuse_strength;

    // vec3 view_dir = normalize(v_view_position - v_position); // UPDATED!
    // vec3 half_dir = normalize(view_dir + light_dir);
    // float specular_strength = pow(max(dot(normal, half_dir), 0.0), 32);
    // vec3 specular_color = specular_strength * light_color;

    // vec3 result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;

    //f_color = vec4(result, object_color.a);


    //f_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords);
    //vec3 living_coral_color = tint;// vec3(1.0, 0.43, 0.38);
    
}