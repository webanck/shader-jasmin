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
//a disk of unit radius in the xy plane
float diskD(in vec3 p)
{
	float r2 = dot(p.xy, p.xy);
	//Projection in the disk, closest is disk surface.
	if(r2 < 1.0) return abs(p.z);
	
	//Projection out of disk, closest is distance to circle.
	float r = sqrt(r2);
	float h = r - 1.0;
	return sqrt(h*h + p.z*p.z);
}
//maturity in [0, 2]
float petal(in float r, in float maturity, in vec3 p)
{
	//a sphere describing the extremity of the petal trajectory (rotation) through maturation
	//vec3 c = vec3(0, 0, r);
	//return sphereD(c, 0.1*r, rotateY(0.5*maturity*PI, p));

	//scale
	p /= r*vec3(1, 0.5, 1.);
	//rotate
	p = rotateY(0.5*maturity*PI, p);
	//shift
	p.x -= 1.;
	return diskD(p);
}
//flower in the xy plane
const uint NB_PETALS = 5u;
float jasmin(in float r, in float maturity, in vec3 p)
{
	//return petal(r, maturity, p);

	float d = 2.*r;
	for(uint i = 0u; i < NB_PETALS; i++)
		d = min(d, petal(r, maturity, rotateZ(float(i)/float(NB_PETALS)*2.*PI, p)));
	return d;
}
float jasminD(in vec3 c, in float r, in vec3 orientation, in float maturity, in vec3 p)
{
	return jasmin(r, maturity, p - c); //todo: p orientation transform
}
const float STEP_SIZE = 0.01;
const float NB_STEPS = 1000u;
const float EPSILON = 1.1*STEP_SIZE;
const float MAX_DEPTH = 10.;
bool intersect(in vec3 ro, in vec3 rd, out vec3 intersection)
{
	float t = 0.;
	for(uint i = 0u; i < NB_STEPS && t < MAX_DEPTH; i++)
	{
		intersection = ro + t*rd;
		
		float radius = 0.15;
		float maturity = 2.*iMouse.x/iResolution.x;
		float d = jasminD(vec3(0, 0, -0.1), radius, vec3(0, 0, 1), maturity, intersection);
		if(d < EPSILON) return true;
		
		t += STEP_SIZE;
		//t += max(STEP_SIZE, d);
	}
	return false;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
	vec3 ro;
	vec3 rd;
	pixelRay(fragCoord.xy, ro, rd);
	vec3 intersection;
	fragColor = vec4(intersect(ro, rd, intersection) ? 0.5 + 0.5*intersection : vec3(0), 1.);
}
