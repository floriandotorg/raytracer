extern crate multiqueue;

use pixel_canvas::{Canvas};
use std::sync::mpsc;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::ops::{Add, Mul};
use cgmath::{Point3, Vector3, dot, prelude::*};
use std::{thread};

type Float = f64;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const FOV: Float = 90.0;

const GAMMA: Float = 2.2;

fn gamma_encode(linear: Float) -> Float {
    linear.powf(1.0 / GAMMA)
}

fn gamma_decode(encoded: Float) -> Float {
    encoded.powf(GAMMA)
}

#[derive(Debug, Copy, Clone)]
pub struct Color {
    pub red: Float,
    pub green: Float,
    pub blue: Float,
}

impl Color {
    pub fn clamp(&self) -> Color {
        Color {
            red: self.red.min(1.0).max(0.0),
            blue: self.blue.min(1.0).max(0.0),
            green: self.green.min(1.0).max(0.0),
        }
    }

    pub fn to_rgba(&self) -> pixel_canvas::Color {
        pixel_canvas::Color {
            r: (gamma_encode(self.red) * 255.0) as u8,
            g: (gamma_encode(self.green) * 255.0) as u8,
            b: (gamma_encode(self.blue) * 255.0) as u8
        }
    }

    pub fn from_rgba(rgba: &pixel_canvas::Color) -> Color {
        Color {
            red: gamma_decode((rgba.r as Float) / 255.0),
            green: gamma_decode((rgba.g as Float) / 255.0),
            blue: gamma_decode((rgba.b as Float) / 255.0),
        }
    }
}
impl Mul for Color {
    type Output = Color;

    fn mul(self, other: Color) -> Color {
        Color {
            red: self.red * other.red,
            blue: self.blue * other.blue,
            green: self.green * other.green,
        }
    }
}
impl Mul<Float> for Color {
    type Output = Color;

    fn mul(self, other: Float) -> Color {
        Color {
            red: self.red * other,
            blue: self.blue * other,
            green: self.green * other,
        }
    }
}
impl Mul<Color> for Float {
    type Output = Color;
    fn mul(self, other: Color) -> Color {
        other * self
    }
}
impl Add for Color {
    type Output = Color;
    fn add(self, other: Color) -> Color {
        Color {
            red: self.red + other.red,
            blue: self.blue + other.blue,
            green: self.green + other.green,
        }
    }
}

type Point = Point3<Float>;
type Vector = Vector3<Float>;

struct Ray {
    origin: Point,
    direction: Vector,
}

trait Node {
    fn intersect(&self, ray: &Ray) -> Option<Float>;
    fn surface_normal(&self, hit_point: &Point) -> Vector;
    fn color(&self) -> Color;
    fn albedo(&self) -> Float;
    fn surface(&self) -> Surface;
}

#[derive(Debug, Clone, Copy)]
enum Surface {
    Diffuse,
    Reflective { reflectivity: Float },
    Refractive { index: Float, transparency: Float },
}

struct Sphere {
    center: Point,
    radius: Float,
    color: Color,
    albedo: Float,
    surface: Surface
}

impl Sphere {
    fn new(center: Point, radius: Float, color: Color, albedo: Float, surface: Surface) -> Self {
        Sphere { center: center, radius: radius, color: color, albedo: albedo, surface: surface }
    }
}

impl Node for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<Float> {
        let l: Vector = self.center - ray.origin;
        let adj = dot(l, ray.direction);
        let d2 = dot(l, l) - (adj * adj);
        let radius2 = self.radius * self.radius;
        if d2 > radius2 {
            return None;
        }
        let thc = (radius2 - d2).sqrt();
        let t0 = adj - thc;
        let t1 = adj + thc;

        if t0 < 0.0 && t1 < 0.0 {
            return None;
        }

        let distance = if t0 < t1 { t0 } else { t1 };
        Some(distance)
    }

    fn surface_normal(&self, hit_point: &Point) -> Vector {
        (*hit_point - self.center).normalize()
    }

    fn color(&self) -> Color {
        self.color
    }

    fn albedo(&self) -> Float {
        self.albedo
    }

    fn surface(&self) -> Surface {
        self.surface
    }
}

struct Plane {
    origin: Point,
    normal: Vector,
    color: Color,
    albedo: Float,
    surface: Surface
}

impl Plane {
    fn new(origin: Point, normal: Vector, color: Color, albedo: Float, surface: Surface ) -> Self {
        Plane { origin: origin, normal: normal, color: color, albedo: albedo, surface: surface }
    }
}

impl Node for Plane {
    fn intersect(&self, ray: &Ray) -> Option<Float> {
        let normal = self.normal;
        let denom = dot(normal, ray.direction);
        if denom > 1e-6 {
            let v = self.origin - ray.origin;
            let distance = dot(v, normal) / denom;
            if distance >= 0.0 {
                return Some(distance);
            }
        }
        None
    }

    fn surface_normal(&self, _: &Point) -> Vector {
        -self.normal
    }

    fn color(&self) -> Color {
        self.color
    }

    fn albedo(&self) -> Float {
        self.albedo
    }

    fn surface(&self) -> Surface {
        self.surface
    }
}

type BoxedNode = Box<dyn Node + Sync>;
type NodeList = Vec<BoxedNode>;

struct DirectionalLight {
    direction: Vector,
    color: Color,
    intensity: Float,
}

struct SphericalLight {
    position: Point,
    color: Color,
    intensity: Float,
}

enum Light {
    Directional(DirectionalLight),
    Spherical(SphericalLight),
}

impl Light {
    fn color(&self) -> Color {
        match *self {
            Light::Directional(ref d) => d.color,
            Light::Spherical(ref s) => s.color,
        }
    }

    fn direction_from(&self, hit_point: &Point) -> Vector {
        match *self {
            Light::Directional(ref d) => -d.direction,
            Light::Spherical(ref s) => (s.position - *hit_point).normalize(),
        }
    }

    fn intensity(&self, hit_point: &Point) -> Float {
        match *self {
            Light::Directional(ref d) => d.intensity,
            Light::Spherical(ref s) => {
                let r2 = (s.position - *hit_point).magnitude() as f64;
                s.intensity / (4.0 * ::std::f64::consts::PI * r2)
            }
        }
    }

    fn distance(&self, hit_point: &Point) -> f64 {
        match *self {
            Light::Directional(_) => ::std::f64::INFINITY,
            Light::Spherical(ref s) => (s.position - *hit_point).magnitude2(),
        }
    }
}

const SHADOW_BIAS: Float = 1e-13;
const BLACK: Color = Color { red: 0.0, green: 0.0, blue: 0.0 };
const MAX_RECURSION_DEPTH: usize = 10;

fn trace<'a>(scene: &'a NodeList, ray: &Ray) -> Option<(Float, &'a BoxedNode)> {
    return scene
        .iter()
        .filter_map(|s| s.intersect(&ray).map(|d| (d, s)))
        .min_by(|i1, i2| i1.0.partial_cmp(&i2.0).unwrap());
}

fn cast_ray(scene: &NodeList, lights: &[Light], ray: &Ray, depth: usize) -> Color {
    if depth >= MAX_RECURSION_DEPTH {
        return BLACK;
    }

    let intersection = trace(scene, &ray);
    return intersection.map(|i| get_color(scene, lights, &ray, i.0, i.1, depth)).unwrap_or(BLACK);
}

fn fresnel(incident: Vector, normal: Vector, index: Float) -> Float {
    let i_dot_n = dot(incident, normal);
    let mut eta_i = 1.0;
    let mut eta_t = index as f64;
    if i_dot_n > 0.0 {
        eta_i = eta_t;
        eta_t = 1.0;
    }

    let sin_t = eta_i / eta_t * (1.0 - i_dot_n * i_dot_n).max(0.0).sqrt();
    if sin_t > 1.0 {
        return 1.0;
    } else {
        let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
        let cos_i = cos_t.abs();
        let r_s = ((eta_t * cos_i) - (eta_i * cos_t)) / ((eta_t * cos_i) + (eta_i * cos_t));
        let r_p = ((eta_i * cos_i) - (eta_t * cos_t)) / ((eta_i * cos_i) + (eta_t * cos_t));
        return (r_s * r_s + r_p * r_p) / 2.0;
    }
}

fn shade_diffuse(scene: &NodeList, lights: &[Light],
    node: &BoxedNode,
    hit_point: Point,
    surface_normal: Vector)
    -> Color {
    let mut color = BLACK;
    for light in lights {
        let direction_to_light = light.direction_from(&hit_point);

        let shadow_ray = Ray {
            origin: hit_point + (surface_normal * SHADOW_BIAS),
            direction: direction_to_light,
        };
        let shadow_intersection = trace(scene, &shadow_ray);
        let in_light = shadow_intersection.is_none() || shadow_intersection.unwrap().0 > light.distance(&hit_point);

        let light_intensity = if in_light {
            light.intensity(&hit_point)
        } else {
            0.0
        };
        let light_power = (dot(surface_normal, direction_to_light) as Float).max(0.0) * light_intensity;
        let light_reflected = node.albedo() / std::f64::consts::PI;

        let light_color = light.color() * light_power * light_reflected;
        color = color + (node.color() * light_color);
    }
    color.clamp()
}

fn create_reflection(normal: Vector,
    incident: Vector,
    intersection: Point,
    bias: Float)
    -> Ray {
    Ray {
        origin: intersection + (normal * bias),
        direction: incident - (2.0 * dot(incident, normal) * normal),
    }
}

fn create_transmission(normal: Vector,
    incident: Vector,
    intersection: Point,
    bias: Float,
    index: Float)
    -> Option<Ray> {
    let mut ref_n = normal;
    let mut eta_t = index as f64;
    let mut eta_i = 1.0;
    let mut i_dot_n = dot(incident, normal);
    if i_dot_n < 0.0 {
        i_dot_n = -i_dot_n;
    } else {
        ref_n = -normal;
        eta_i = eta_t;
        eta_t = 1.0;
    }
    let eta = eta_i / eta_t;
    let k = 1.0 - (eta * eta) * (1.0 - i_dot_n * i_dot_n);
    if k < 0.0 {
        None
    } else {
        Some(Ray {
            origin: intersection + (ref_n * -bias),
            direction: (incident + i_dot_n * ref_n) * eta - ref_n * k.sqrt(),
        })
    }
}

fn get_color(scene: &NodeList, lights: &[Light], ray: &Ray, distance: Float, node: &BoxedNode, depth: usize) -> Color {
    let hit = ray.origin + (ray.direction * distance);
    let normal = node.surface_normal(&hit);

    match node.surface() {
        Surface::Diffuse => shade_diffuse(scene, lights, node, hit, normal),
        Surface::Reflective { reflectivity } => {
            let mut color = shade_diffuse(scene, lights, node, hit, normal);
            let reflection_ray =
                create_reflection(normal, ray.direction, hit, SHADOW_BIAS);
            color = color * (1.0 - reflectivity);
            color = color + (cast_ray(scene, lights, &reflection_ray, depth + 1) * reflectivity);
            color
        }
        Surface::Refractive { index, transparency } => {
            let mut refraction_color = BLACK;
            let kr = fresnel(ray.direction, normal, index) as Float;
            let surface_color = node.color();

            if kr < 1.0 {
                let transmission_ray = create_transmission(normal, ray.direction, hit, SHADOW_BIAS, index).unwrap();
                refraction_color = cast_ray(scene, lights, &transmission_ray, depth + 1);
            }

            let reflection_ray = create_reflection(normal, ray.direction, hit, SHADOW_BIAS);
            let reflection_color = cast_ray(scene, lights, &reflection_ray, depth + 1);
            let mut color = reflection_color * kr + refraction_color * (1.0 - kr);
            color = color * transparency * surface_color;
            color
        }
    }
}

fn raytrace(scene: &NodeList, lights: &[Light], x: usize, y: usize) -> Color {
    assert!(WIDTH >= HEIGHT);
    let fov_adjustment = (FOV.to_radians() / 2.0).tan();
    let aspect_ratio = (WIDTH as Float) / (HEIGHT as Float);
    let sensor_x = ((((x as Float + 0.5) / WIDTH as Float) * 2.0 - 1.0) * aspect_ratio) *
        fov_adjustment;
    let sensor_y = (1.0 - ((y as Float + 0.5) / HEIGHT as Float) * 2.0) * fov_adjustment;

    let ray  = Ray {
        origin: Point::new(0.0, 0.0, 0.0),
        direction: Vector3 {
                x: sensor_x,
                y: sensor_y,
                z: -1.0,
            }
            .normalize(),
    };

    let intersection = trace(scene, &ray);

    match intersection {
        Some((distance, node)) => get_color(scene, &lights, &ray, distance, node, 0),
        _ => BLACK
    }
}


fn main() {
    let (tx, rx) = mpsc::channel();
    let (ttx, rrx) = multiqueue::mpmc_queue((HEIGHT*WIDTH) as u64);

    {
        let mut arr = Vec::with_capacity(WIDTH * HEIGHT);
        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                arr.push((x, y));
            }
        }
        let mut rng = thread_rng();
        arr.shuffle(&mut rng);
        for (_, v) in arr.iter().enumerate() {
            ttx.try_send(v.clone()).unwrap();
        }
    }

    for _ in 0..(num_cpus::get()*2) {
        let tx = tx.clone();
        let rrx = rrx.clone();
        thread::spawn(|| {
            let tx = tx;
            let rrx = rrx;
            let scene: NodeList = vec![
                Box::new(Sphere::new(Point::new(0.0, 0.0, -5.0), 1.0, Color { red: 1.0, green: 1.0, blue: 1.0 }, 0.18, Surface::Refractive { index: 1.5, transparency: 0.8 })),
                Box::new(Sphere::new(Point::new(-3.0, 1.0, -6.0), 2.0, Color { red: 0.1, green: 0.1, blue: 0.8 }, 0.58, Surface::Diffuse)),
                Box::new(Sphere::new(Point::new(2.0, 1.0, -4.0), 2.0, Color { red: 0.2, green: 1.0, blue: 0.2 }, 0.18, Surface::Reflective { reflectivity: 0.8 })),
                Box::new(Plane::new(Point::new(0.0, -2.0, -5.0), Vector::new(0.0, -1.0, 0.0), Color { red: 0.2, green: 0.2, blue: 0.2 }, 0.18, Surface::Reflective { reflectivity: 0.5 })),
                Box::new(Plane::new(Point::new(0.0, -0.0, -100.0), Vector::new(0.0, 0.0, -1.0), Color { red: 0.2, green: 0.3, blue: 1.0 }, 0.38, Surface::Diffuse))
            ];
            let lights = [
                Light::Directional(DirectionalLight { color: Color { red: 1.0, green: 1.0, blue: 1.0 }, intensity: 1.0, direction: Vector::new(-0.25, -1.0, -1.0) }),
                Light::Spherical(SphericalLight { color: Color { red: 0.8, green: 0.3, blue: 0.3 }, position: Point::new(0.25, 0.0, -2.0), intensity: 250.0 }),
                Light::Spherical(SphericalLight { color: Color { red: 0.3, green: 0.8, blue: 0.3 }, position: Point::new(-2.0, 10.0, -7.0), intensity: 10000.0 })
            ];
            loop {
                match rrx.try_recv() {
                    Ok((x, y)) => {
                        let color = raytrace(&scene, &lights, x, y);
                        tx.send((x, y, color)).unwrap();
                    }
                    Err(_) => break
                }
            }
        });
    }

    let mut pixels: Vec<Vec<Option<pixel_canvas::Color>>> = vec![vec![None; WIDTH]; HEIGHT];
    let canvas = Canvas::new(WIDTH, HEIGHT).title("Raytracer");
    canvas.render(move |_, image| {
        loop {
            match rx.try_recv() {
                Ok((x, y, color)) => pixels[y][x] = Some(color.to_rgba()),
                Err(_) => break
            }
        }

        let width = image.width() as usize;
        for (y, row) in image.chunks_mut(width).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                match pixels[HEIGHT - y - 1][x] {
                    Some(color) => *pixel = color,
                    None => *pixel = pixel_canvas::Color{
                        r: 255,
                        g: 255,
                        b: 255
                    }
                }
            }
        }
    });
}
