#define EXPOSURE_FACTOR 2.
#define GAMMA_FACTOR 1.3

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 pixelUV = fragCoord/iResolution.xy;
	vec4 pixelValue = texture(iChannel0, pixelUV);
	
	if(EXPOSURE_FACTOR > 0.0)
		pixelValue.xyz = vec3(1.0) - exp(-pixelValue.xyz * EXPOSURE_FACTOR);
		
	pixelValue.xyz = pow(pixelValue.xyz, vec3(GAMMA_FACTOR));
	fragColor = pixelValue;
}
