const float PI = 3.14;

float obsDistance(const uint obsId)
{
	//Inside view.
	if(obsId == 0u)
		return 1.;
	//Outside view.
	else
		return 8.;
}
mat3 observerViewMatrix(const uint obsId, vec2 mouseUV)
{
	vec2 shift = vec2(0.);
	float a = (shift.x + mouseUV.x)*PI*2.;
	float b = (shift.y + mouseUV.y)*PI;
	//if (mouseUV == vec2(0)) {b = -0.2; a = 0.2;}
	//if(obsId == 1u)
	//{
	//	a *= -1.;
	//	b *= -1.;
	//}
	
	vec3 camera = vec3(cos(b)*sin(a), sin(b), cos(b)*cos(a));
	//Z vector
	vec3 up = normalize(cross(cross(camera, vec3(0, 1, 0)), camera));
	//Y vector
	vec3 x = normalize(cross(up, camera));
	
	//inside view
	return obsDistance(obsId)*mat3(x, up, camera);
}
void pixelRay(in vec2 ij, out vec3 ro, out vec3 rd)
{
	//Towards -Z.
	ro = vec3(0, 0, 1);
	vec2 q = (ij - 0.5*iResolution.xy)/iResolution.y;
	rd = normalize(vec3(q, 0) - ro);

	//mat3 view = observerViewMatrix(0u, iMouse.xy/iResolution.xy - 0.5);
	//ro = view[2];
	//rd = normalize(view*rd);
}

vec3 rotateZ(in float rads, in vec3 p)
{
	return vec3(
		cos(rads)*p.x - sin(rads)*p.y,
		sin(rads)*p.x + cos(rads)*p.y,
		p.z
	);
}
vec3 rotateX(in float rads, in vec3 p)
{
	return rotateZ(rads, p.yzx).zxy;
}
vec3 rotateY(in float rads, in vec3 p)
{
	return rotateZ(rads, p.zxy).yzx;
}

float sphereD(in vec3 c, in float r, in vec3 p)
{
	return length(p - c) - r;
}
//maturity in [0, 2]
float petal(in float r, in float maturity, in vec3 p)
{
	vec3 c = vec3(0, 0, r);
	return sphereD(c, 0.1*r, rotateY(0.5*maturity*PI, p));
}
const uint NB_PETALS = 5u;
float jasmin(in float r, in float maturity, in vec3 p)
{
	float d = 2.*r;
	for(uint i = 0u; i < NB_PETALS; i++)
		d = min(d, petal(r, maturity, rotateZ(float(i)/float(NB_PETALS)*2.*PI, p)));
	return d;
}
float jasminD(in vec3 c, in float r, in vec3 orientation, in float maturity, in vec3 p)
{
	return jasmin(r, maturity, p - c); //todo: p orientation transform
}
bool intersect(in vec3 ro, in vec3 rd, out vec3 intersection)
{
	intersection = ro;
	for(uint i = 0u; i < 1000u; i++)
	{
		float radius = 0.5;
		float maturity = 2.*iMouse.x/iResolution.x;
		float d = jasminD(vec3(0, 0, -2), radius, vec3(0, 0, 1), maturity, intersection);
		if(d < 0) return true;
		
		intersection += rd*0.01;
	}
	return false;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
	vec3 ro;
	vec3 rd;
	pixelRay(fragCoord.xy, ro, rd);
	vec3 intersection;
	fragColor = vec4(intersect(ro, rd, intersection) ? -0.5*intersection : vec3(0), 1.);
}
