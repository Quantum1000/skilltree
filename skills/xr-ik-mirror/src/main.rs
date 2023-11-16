use std::f32::consts::PI;
use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::EulerRot::XYZ;
use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::render_resource::{
	Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::transform::components::Transform;

use bevy_openxr::input::XrInput;
use bevy_openxr::resources::XrFrameState;
use bevy_openxr::xr_input::oculus_touch::OculusController;
use bevy_openxr::xr_input::{QuatConv, Vec3Conv};
use bevy_openxr::DefaultXrPlugins;

use std::collections::HashMap;

use color_eyre::eyre;


const ASSET_FOLDER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/");
// Much of the IK Algorithm is sourced from [1]: doi:10.1145/3281505.3281529
// units in meters and radians
const HEAD_HEIGHT_FACTOR: f32 = 1.15; // a factor for converting from the height of the avatar's head bone transform to the top of their head
const DEFAULT_HEAD_HEIGHT: f32 = 1.69; // slightly more than the height of the average American (men and women)
const HEURISTIC_NECK_PITCH_SCALE_CONSTANT: f32 = 0.7517 * PI; // ~135.3 degrees, from [1]
const HEURISTIC_NECK_PITCH_HEAD_CONSTANT: f32 = 0.333; // from [1]
const HEURISTIC_CHEST_PITCH_HEAD_CONSTANT: f32 = 0.05; // I made this up, based on observation that the chest responded less to head tilt
const HEURISTIC_SHOULDER_SCALING_CONSTANT: f32 = 0.167 * PI; // from [1]
const DEFAULT_SHOULDER_MOVEMENT_THRESHOLD: f32 = 0.5; // from [1]
const DEFAULT_SHOULDER_ROTATION_YAW_CONSTRAINT_MIN: f32 = 0.; // from [1], probably wouldn't hurt if it was a few degrees lower.
const DEFAULT_SHOULDER_ROTATION_YAW_CONSTRAINT_MAX: f32 = 0.183 * PI; // ~33 degrees, from [1]. Seems about right.
const DEFAULT_SHOULDER_ROTATION_ROLL_CONSTRAINT_MIN: f32 = 0.; // from [1], doesn't seem right these are the same
const DEFAULT_SHOULDER_ROTATION_ROLL_CONSTRAINT_MAX: f32 = 0.183 * PI; // from [1], doesn't seem right these are the same
const HEURISTIC_ELBOW_MODEL_BIASES: Vec3 = Vec3 {x: 30., y: 120., z: 65.}; // from [1], derived through manual optimization
const HEURISTIC_ELBOW_MODEL_WEIGHTS: Vec3 = Vec3 {x: -50., y: -60., z: 260.}; // from [1], derived through manual optimization
const HEURISTIC_ELBOW_MODEL_OFFSET: f32 = 0.083 * PI; // ~15 degrees, from [1]
const HEURISTIC_ELBOW_MODEL_CONSTRAINT_MIN: f32 = 0.072 * PI; // ~13 degrees, from [1]
const HEURISTIC_ELBOW_MODEL_CONSTRAINT_MAX: f32 = 0.972 * PI; // ~175 degrees, from [1]
const HEURISTIC_ELBOW_SINGULARITY_RADIAL_THRESHOLD: f32 = 0.5; // from [1]
const HEURISTIC_ELBOW_SINGULARITY_FORWARD_THRESHOLD_MIN: f32 = 0.; // from [1]
const HEURISTIC_ELBOW_SINGULARITY_FORWARD_THRESHOLD_MAX: f32 = 0.1; // from [1]
const HEURISTIC_ELBOW_SINGULARITY_VECTOR: Vec3 = Vec3 {x: 0.133, y: -0.443, z: -0.886}; // from [1], probably normalized but I haven't checked; represents a direction.
// to some extent it feels like the wrist yaw heuristics should come with constraints
const HEURISTIC_ELBOW_WRIST_YAW_THRESHOLD_LOWER: f32 = -0.25 * PI; // 45 degrees, from [1]
const HEURISTIC_ELBOW_WRIST_YAW_THRESHOLD_UPPER: f32 = 0.25 * PI; // from [1], mostly makes sense that these are the same.
const HEURISTIC_ELBOW_WRIST_YAW_SCALING_CONSTANT_LOWER: f32 = 1./(-0.75 * PI); // 135^-1 degrees, from [1]. It makes sense in context.
const HEURISTIC_ELBOW_WRIST_YAW_SCALING_CONSTANT_UPPER: f32 = 1./(0.75 * PI); // 135^-1 degrees, from [1]
const HEURISTIC_ELBOW_WRIST_ROLL_THRESHOLD_LOWER: f32 = 0.; // 0 degrees, from [1].
const HEURISTIC_ELBOW_WRIST_ROLL_THRESHOLD_UPPER: f32 = 0.5 * PI; // from [1]
const HEURISTIC_ELBOW_WRIST_ROLL_SCALING_CONSTANT_LOWER: f32 = 1./(-(1./0.3) * PI); // 600^-1 degrees, from [1]
const HEURISTIC_ELBOW_WRIST_ROLL_SCALING_CONSTANT_UPPER: f32 = 1./(1./0.6 * PI); // 300^-1 degrees, from [1]


fn main() {
	color_eyre::install().unwrap();

	info!("Running `openxr-6dof` skill");
	App::new()
		.add_plugins(DefaultXrPlugins)
		.add_plugins(LogDiagnosticsPlugin::default())
		.add_plugins(FrameTimeDiagnosticsPlugin)
		.add_plugins(bevy_mod_inverse_kinematics::InverseKinematicsPlugin)
		.add_systems(Startup, setup)
		.add_systems(
			Update,
			(update_ik, setup_ik),
		)
		.run();
}

#[derive(Component)]
pub struct AvatarSetup;

/// set up a simple 3D scene
fn setup(
	mut commands: Commands,
	mut meshes: ResMut<Assets<Mesh>>,
	mut images: ResMut<Assets<Image>>,
	assets: Res<AssetServer>,
	mut materials: ResMut<Assets<StandardMaterial>>,
) {
	let bevy_mirror_dwelling_img: Handle<Image> =
		assets.load(&(ASSET_FOLDER.to_string() + "bevy_mirror_dwelling.png"));
	commands.spawn(PbrBundle {
		mesh: meshes.add(shape::Cube::default().into()),
		material: materials.add(StandardMaterial {
			base_color_texture: Some(bevy_mirror_dwelling_img),
			..default()
		}),
		transform: Transform::from_xyz(0.0, 2.2, -2.0)
			.with_scale(Vec3::new(2.0, 0.5, 0.01))
			.with_rotation(Quat::from_euler(
				XYZ,
				180.0_f32.to_radians(),
				0.0,
				180.0_f32.to_radians(),
			)),
		..default()
	});
	let size = Extent3d {
		width: 512,
		height: 512,
		..default()
	};
	// This is the texture that will be rendered to.
	let mut image = Image {
		texture_descriptor: TextureDescriptor {
			label: None,
			size,
			dimension: TextureDimension::D2,
			format: TextureFormat::Bgra8UnormSrgb,
			mip_level_count: 1,
			sample_count: 1,
			usage: TextureUsages::TEXTURE_BINDING
				| TextureUsages::COPY_DST
				| TextureUsages::RENDER_ATTACHMENT,
			view_formats: &[],
		},
		..default()
	};

	// fill image.data with zeroes
	image.resize(size);
	// image for the mirror
	let image_handle = images.add(image);

	// material for the mirror
	let mirror_material_handle = materials.add(StandardMaterial {
		base_color_texture: Some(image_handle.clone()),
		reflectance: 0.02,
		unlit: false,
		..default()
	});
	// the plane displaying the mirrors texture
	let mirror = commands
		.spawn(PbrBundle {
			mesh: meshes.add(Mesh::from(shape::Quad {
				size: Vec2 { x: 2.0, y: 2.0 },
				flip: true,
			})),
			material: mirror_material_handle,
			..default()
		})
		.id();
	// camera for mirror
	let camera = commands
		.spawn(Camera3dBundle {
			camera_3d: Camera3d {
				clear_color: ClearColorConfig::Custom(Color::WHITE),
				..default()
			},
			camera: Camera {
				// render before the "main pass" camera
				order: -1,
				target: RenderTarget::Image(image_handle.clone()),
				..default()
			},
			transform: Transform::from_xyz(0.0, 0.0, 0.0).with_rotation(
				Quat::from_euler(EulerRot::XYZ, 0.0, 180.0_f32.to_radians(), 0.0),
			),
			..default()
		})
		.id();
	commands
		.spawn(SpatialBundle {
			transform: Transform::from_xyz(0.0, 1.0, -2.0),
			..default()
		})
		.push_children(&[camera, mirror]);

	// plane
	commands.spawn(PbrBundle {
		mesh: meshes.add(shape::Plane::from_size(5.0).into()),
		material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
		..default()
	});
	// cube
	commands.spawn(PbrBundle {
		mesh: meshes.add(Mesh::from(shape::Cube { size: 0.1 })),
		material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
		transform: Transform::from_xyz(0.0, 0.5, 0.0),
		..default()
	});
	// light
	commands.spawn(PointLightBundle {
		point_light: PointLight {
			intensity: 1500.0,
			shadows_enabled: true,
			..default()
		},
		transform: Transform::from_xyz(4.0, 8.0, 4.0),
		..default()
	});
	// camera
	commands.spawn((Camera3dBundle {
		transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
		..default()
	},));
	commands.spawn((
		SceneBundle {
			scene: assets.load(&(ASSET_FOLDER.to_string() + "/malek.gltf#Scene0")),
			transform: Transform::from_xyz(0.0, 0.0, 0.0).with_rotation(
				Quat::from_euler(EulerRot::XYZ, 0.0, 0.0_f32.to_radians(), 0.0),
			),
			..default()
		},
		AvatarSetup,
	));
}


#[derive(Component)]
pub struct Bone;

#[derive(Component)]
pub struct TrueHead;

#[derive(Component)]
pub struct SkeletonComponent {
	entities: [Option<Entity>; SKELETON_ARR_LEN],
	height: f32,
	defaults: Skeleton,
}

const HEAD: usize = 0;
const HIPS: usize = 1;
const SPINE: usize = 2;
const CHEST: usize = 3;
const UPPER_CHEST: usize = 4;
const NECK: usize = 5;
const LEFT_SHOULDER: usize = 6;
const LEFT_EYE: usize = 7;
const LEFT_FOOT: usize = 8;
const LEFT_HAND: usize = 9;
const LEFT_LEG_UPPER: usize = 10;
const LEFT_LEG_LOWER: usize = 11;
const LEFT_ARM_UPPER: usize = 12;
const LEFT_ARM_LOWER: usize = 13;
const RIGHT_SHOULDER: usize = 14;
const RIGHT_EYE: usize = 15;
const RIGHT_FOOT: usize = 16;
const RIGHT_HAND: usize = 17;
const RIGHT_LEG_UPPER: usize = 18;
const RIGHT_LEG_LOWER: usize = 19;
const RIGHT_ARM_UPPER: usize = 20;
const RIGHT_ARM_LOWER: usize = 21;
const SKELETON_ARR_LEN: usize = 23;

#[derive(Copy, Clone)]
struct Skeleton{
	head: Transform,
	hips: Transform,
	spine: Transform,
	chest: Option<Transform>,
	upper_chest: Option<Transform>,
	neck: Option<Transform>,
	left: SkeletonSide,
	right: SkeletonSide
}

#[derive(Copy, Clone)]
struct SkeletonSide {
	shoulder: Option<Transform>,
	eye: Option<Transform>,
	leg: Limb,
	arm: Limb,
	foot: Transform,
	hand: Transform,
}

#[derive(Copy, Clone)]
struct Limb {
	upper: Transform,
	lower: Transform
}

impl Skeleton {
	pub fn new(skeleton: [Option<Transform>; SKELETON_ARR_LEN]) -> color_eyre::Result<Self> {
		Ok(Self {
			head: skeleton[HEAD].ok_or(eyre::Report::msg("bad Skeleton"))?,
			hips: skeleton[HIPS].ok_or(eyre::Report::msg("bad Skeleton"))?,
			spine: skeleton[SPINE].ok_or(eyre::Report::msg("bad Skeleton"))?,
			chest: skeleton[CHEST],
			upper_chest: skeleton[UPPER_CHEST],
			neck: skeleton[NECK],
			left: SkeletonSide {
				shoulder: skeleton[LEFT_SHOULDER],
				eye: skeleton[LEFT_EYE],
				foot: skeleton[LEFT_FOOT].ok_or(eyre::Report::msg("bad Skeleton"))?,
				hand: skeleton[LEFT_HAND].ok_or(eyre::Report::msg("bad Skeleton"))?,
				leg: Limb {
					upper: skeleton[LEFT_LEG_UPPER].ok_or(eyre::Report::msg("bad Skeleton"))?,
					lower: skeleton[LEFT_LEG_LOWER].ok_or(eyre::Report::msg("bad Skeleton"))?,
				},
				arm: Limb {
					upper: skeleton[LEFT_ARM_UPPER].ok_or(eyre::Report::msg("bad Skeleton"))?,
					lower: skeleton[LEFT_LEG_LOWER].ok_or(eyre::Report::msg("bad Skeleton"))?,
				},
			},
			right: SkeletonSide {
				shoulder: skeleton[RIGHT_SHOULDER],
				eye: skeleton[RIGHT_EYE],
				foot: skeleton[RIGHT_FOOT].ok_or(eyre::Report::msg("bad Skeleton"))?,
				hand: skeleton[RIGHT_HAND].ok_or(eyre::Report::msg("bad Skeleton"))?,
				leg: Limb {
					upper: skeleton[RIGHT_LEG_UPPER].ok_or(eyre::Report::msg("bad Skeleton"))?,
					lower: skeleton[RIGHT_LEG_LOWER].ok_or(eyre::Report::msg("bad Skeleton"))?,
				},
				arm: Limb {
					upper: skeleton[RIGHT_ARM_UPPER].ok_or(eyre::Report::msg("bad Skeleton"))?,
					lower: skeleton[RIGHT_ARM_LOWER].ok_or(eyre::Report::msg("bad Skeleton"))?,
				},
			}
		})
	}
	pub fn arrayify(self) -> [Option<Transform>; SKELETON_ARR_LEN]
	{
		let mut skeleton: [Option<Transform>; SKELETON_ARR_LEN] = Default::default();
		skeleton[HEAD] = Some(self.head);
		skeleton[HIPS] = Some(self.hips);
		skeleton[SPINE] = Some(self.spine);
		skeleton[CHEST] = self.chest;
		skeleton[UPPER_CHEST] = self.upper_chest;
		skeleton[NECK] = self.neck;
		skeleton[LEFT_SHOULDER] = self.left.shoulder;
		skeleton[LEFT_EYE] = self.left.eye;
		skeleton[LEFT_FOOT] = Some(self.left.foot);
		skeleton[LEFT_HAND] = Some(self.left.hand);
		skeleton[LEFT_ARM_UPPER] = Some(self.left.arm.upper);
		skeleton[LEFT_ARM_LOWER] = Some(self.left.arm.lower);
		skeleton[LEFT_LEG_UPPER] = Some(self.left.leg.upper);
		skeleton[LEFT_LEG_LOWER] = Some(self.left.leg.lower);
		skeleton[RIGHT_SHOULDER] = self.right.shoulder;
		skeleton[RIGHT_EYE] = self.right.eye;
		skeleton[RIGHT_FOOT] = Some(self.right.foot);
		skeleton[RIGHT_HAND] = Some(self.right.hand);
		skeleton[RIGHT_ARM_UPPER] = Some(self.right.arm.upper);
		skeleton[RIGHT_ARM_LOWER] = Some(self.right.arm.lower);
		skeleton[RIGHT_LEG_UPPER] = Some(self.right.leg.upper);
		skeleton[RIGHT_LEG_LOWER] = Some(self.right.leg.lower);
		skeleton
	}
	pub fn recalculate_root(mut self) -> Self{
		self.spine = self.spine.mul_transform(self.hips);
		let mut highest_root = self.spine;
		match self.chest {
			Some(t) => {
				let result = t.mul_transform(self.spine);
				self.chest = Some(result);
				highest_root = result;
				match self.upper_chest {
					Some(t) => {
						let result = t.mul_transform(highest_root);
						self.upper_chest = Some(result);
						highest_root = result;
					},
					None => {}
				}
			},
			None => {}
		}
		let mut head_root = highest_root;
		match self.neck {
			Some(t) => {
				let result = t.mul_transform(highest_root);
				self.neck = Some(result);
				head_root = result;
			},
			None => {}
		}
		self.head = self.head.mul_transform(head_root);
		self.left = self.left.recalculate_root(self.head, self.hips, highest_root);
		self.right = self.right.recalculate_root(self.head, self.hips, highest_root);
		self
	}
}

impl SkeletonSide {
	pub fn recalculate_root(mut self, head: Transform, hips: Transform, mut highest_root: Transform) -> Self {
		match self.eye {
			Some(t) => self.eye = Some(t.mul_transform(head)),
			None => {}
		}
		match self.shoulder {
			Some(t) => {
				let result = t.mul_transform(highest_root);
				self.shoulder = Some(result);
				highest_root = result;
			},
			None => {},
		}
		self.arm = self.arm.recalculate_root(highest_root);
		self.leg = self.leg.recalculate_root(hips);
		self.hand = self.hand.mul_transform(self.arm.lower);
		self.foot = self.foot.mul_transform(self.leg.lower);
		self
	}
}

impl Limb {
	pub fn recalculate_root(mut self, root: Transform) -> Self {
		self.upper = self.upper.mul_transform(root);
		self.lower = self.lower.mul_transform(self.upper);
		self
	}
}



fn update_ik(
	skeleton_query: Query<(&GlobalTransform, &SkeletonComponent)>,
	mut transforms: Query<(&mut Transform, &GlobalTransform, With<Bone>)>,
	oculus_controller: Res<OculusController>,
	frame_state: Res<XrFrameState>,
	xr_input: Res<XrInput>,
) {
	let mut func = || -> color_eyre::Result<()> {
		let skeleton_query_out = skeleton_query.single();
		let root = skeleton_query_out.0;
		let skeleton_comp = skeleton_query_out.1;
		let height_factor = skeleton_comp.height / DEFAULT_HEAD_HEIGHT;
		let frame_state = *frame_state.lock().unwrap();
		let headset_input = xr_input
			.head
			.relate(&xr_input.stage, frame_state.predicted_display_time);
		let right_controller_input = oculus_controller
			.grip_space
			.right
			.relate(&xr_input.stage, frame_state.predicted_display_time);
		let left_controller_input = oculus_controller
			.grip_space
			.left
			.relate(&xr_input.stage, frame_state.predicted_display_time);
		// read the state of the skeleton from the transforms
		let mut skeleton_transform_array: [Option<Transform>; SKELETON_ARR_LEN] = Default::default();
		let mut skeleton_root_array: [Option<Transform>; SKELETON_ARR_LEN] = Default::default();
		for i in 0..SKELETON_ARR_LEN {
			let entity = match skeleton_comp.entities[i] {Some(e) => e, None => continue};
			skeleton_transform_array[i] = Some(*transforms.get(entity)?.0);
			skeleton_root_array[i] = Some(transforms.get(entity)?.1.reparented_to(root));
		}
		let mut skeleton = Skeleton::new(skeleton_transform_array)?;
		let mut root_skeleton = Skeleton::new(skeleton_root_array)?;
		// At this point, the skeleton is known to be valid; next, handle VR input
		let final_head = match headset_input {
			Ok(input) => Transform {
				translation: input.0.pose.position.to_vec3()*height_factor,
				rotation: input.0.pose.orientation.to_quat(),
				scale: skeleton.head.scale,
			},
			Err(_) => root_skeleton.head
		};
		let final_left_hand = match left_controller_input {
			Ok(input) => Transform {
				translation: input.0.pose.position.to_vec3()*height_factor,
				rotation: input.0.pose.orientation.to_quat(),
				scale: skeleton.left.hand.scale,
			},
			Err(_) => root_skeleton.left.hand
		};
		let final_right_hand = match right_controller_input {
			Ok(input) => Transform {
				translation: input.0.pose.position.to_vec3()*height_factor,
				rotation: input.0.pose.orientation.to_quat(),
				scale: skeleton.right.hand.scale,
			},
			Err(_) => root_skeleton.right.hand
		};
		// note: solves for the angle. a, b and c should be length squared
		fn cos_law(a: f32, b:f32, c:f32) -> f32 {
			// a+b>c must be true
			((a+b-c) / (2. * a.sqrt() * b.sqrt())).acos()
		}
		// now everything is set up and IK logic can begin.
		let mut chest_pitch = (skeleton_comp.height - final_head.translation.y) / skeleton_comp.height;
		let mut neck_pitch = chest_pitch * (HEURISTIC_NECK_PITCH_SCALE_CONSTANT + HEURISTIC_NECK_PITCH_HEAD_CONSTANT * final_head.rotation.to_euler(EulerRot::YXZ).1);
		chest_pitch *= HEURISTIC_NECK_PITCH_SCALE_CONSTANT + HEURISTIC_CHEST_PITCH_HEAD_CONSTANT * final_head.rotation.to_euler(EulerRot::YXZ).1;
		neck_pitch -= chest_pitch; // Rotations propagate through transforms
		let l_hand_flat = Vec2::new(final_left_hand.translation.x, final_left_hand.translation.z);
		let r_hand_flat = Vec2::new(final_right_hand.translation.x, final_right_hand.translation.z);
		let mut hand_dir = l_hand_flat + r_hand_flat;
		let head_rot = final_head.forward();
		let head_rot_flat = Vec2::new(head_rot.x, head_rot.z);
		if hand_dir.angle_between(head_rot_flat).abs() > 90./180. * PI {
			hand_dir *= -1.;
		}
		let chest_yaw = (hand_dir).angle_between(Vec2::X);
		skeleton.hips.rotation = Quat::from_euler(EulerRot::YXZ, chest_yaw, chest_pitch, 0.);
		// more logic needs to be moved in here, this is currently inefficient when the model is underactuated
		match skeleton.neck {
			Some(mut t) => {
				let neck_rot = Quat::from_euler(EulerRot::YXZ, 0., neck_pitch, 0.);
				t.rotation = neck_rot;
			},
			None => {}
		}
		let final_hands = [final_left_hand, final_right_hand];
		let mut skeleton_sides = [&mut skeleton.left,&mut  skeleton.right];
		let default_skel_sides = [skeleton_comp.defaults.left, skeleton_comp.defaults.right];
		// return shoulders to neutral position
		for i in 0..2 {
			match skeleton_sides[i].shoulder {
				Some(mut t) => t.rotation = default_skel_sides[i].shoulder.ok_or(eyre::Report::msg("wtf how, Skeleton modified in ways it shouldn't be"))?.rotation,
				None => {}
			}
		}
		// calculate position of shoulders and head relative to root
		drop(skeleton_sides);
		root_skeleton = skeleton.recalculate_root(); // this function is likely quite expensive compared to everything else
		let new_offset = final_head.translation - root_skeleton.head.translation;
		skeleton.hips.translation = skeleton.hips.translation + new_offset;
		skeleton.head.rotation = final_head.rotation * root_skeleton.head.rotation.conjugate();
		skeleton_sides = [&mut skeleton.left, &mut  skeleton.right];
		let mut root_skel_sides = [root_skeleton.left, root_skeleton.right];
		let mut had_shoulders = false;
		for i in 0..2 {
			// the spine is assumed to be very close to the center of mass
			let target_foot_z = root_skeleton.spine.translation.z;
			// accounting for scaling is really annoying
			let upper_leg_length = (root_skel_sides[i].leg.lower.translation-root_skel_sides[i].leg.upper.translation).length_squared();
			let lower_leg_length = (root_skel_sides[i].foot.translation-root_skel_sides[i].leg.lower.translation).length_squared();
			let target_foot_pos = default_skel_sides[i].foot.mul_transform(default_skel_sides[i].leg.lower.mul_transform(
				default_skel_sides[i].leg.upper.mul_transform(skeleton_comp.defaults.hips)
			)).translation * Vec3::new(1., 1., 0.) + target_foot_z * Vec3::Z;
			let foot_hip_vec: Vec3 = target_foot_pos - root_skel_sides[i].leg.upper.translation;
			let foot_hip_dist = foot_hip_vec.length_squared();
			let foot_knee_angle = cos_law(upper_leg_length, foot_hip_dist, lower_leg_length);
			let target_knee_pos = Quat::from_axis_angle(Vec3::NEG_Z.cross(foot_hip_vec), foot_knee_angle).mul_vec3(foot_hip_vec).normalize()*upper_leg_length;
			let curr_knee_rel_u_leg = GlobalTransform::from(root_skel_sides[i].leg.lower).reparented_to(&GlobalTransform::from(root_skel_sides[i].leg.upper)).translation;
			let knee_corrective_rot = Quat::from_rotation_arc(curr_knee_rel_u_leg, target_knee_pos);
			skeleton_sides[i].leg.upper.rotate(knee_corrective_rot);
			let updated_root_upper_leg = root_skel_sides[i].leg.upper.with_rotation(root_skel_sides[i].leg.upper.rotation * knee_corrective_rot);
			let curr_root_lower_leg = skeleton_sides[i].leg.lower.mul_transform(updated_root_upper_leg);
			let target_foot_rel_knee = GlobalTransform::from(root_skel_sides[i].foot.with_translation(target_foot_pos)).reparented_to(&GlobalTransform::from(curr_root_lower_leg));
			let foot_corrective_rot = Quat::from_rotation_arc(skeleton_sides[i].foot.translation, target_foot_rel_knee.translation);
			skeleton_sides[i].leg.lower.rotate(foot_corrective_rot);
			let updated_root_lower_leg = curr_root_lower_leg.with_rotation(root_skel_sides[i].leg.lower.rotation * foot_corrective_rot);
			let curr_root_foot = skeleton_sides[i].foot.mul_transform(updated_root_lower_leg);
			skeleton_sides[i].hand.rotate(curr_root_foot.rotation.conjugate() * skeleton_sides[i].foot.rotation);
			match skeleton_sides[i].shoulder {
				Some(mut t) => {
					let shoulder_hand = root_skel_sides[i].arm.upper.translation + new_offset - final_hands[i].translation;
					let calc_shoulder_angle = |input| -> f32 {
						HEURISTIC_SHOULDER_SCALING_CONSTANT * (shoulder_hand.dot(input)/(root_skel_sides[i].arm.lower.translation.length() 
							+ root_skel_sides[i].hand.translation.length()) - DEFAULT_SHOULDER_MOVEMENT_THRESHOLD)
					};
					let mut shoulder_yaw = calc_shoulder_angle(skeleton.spine.forward());
					shoulder_yaw = shoulder_yaw.clamp(DEFAULT_SHOULDER_ROTATION_YAW_CONSTRAINT_MIN, DEFAULT_SHOULDER_ROTATION_YAW_CONSTRAINT_MAX);
					let mut shoulder_roll = calc_shoulder_angle(skeleton.spine.up());
					shoulder_roll = shoulder_roll.clamp(DEFAULT_SHOULDER_ROTATION_ROLL_CONSTRAINT_MIN, DEFAULT_SHOULDER_ROTATION_ROLL_CONSTRAINT_MAX);
					let shoulder_rot = Quat::from_euler(EulerRot::YXZ, shoulder_yaw, 0., shoulder_roll);
					t.rotation = shoulder_rot;
					had_shoulders = true;
				},
				None => {}
			};
		}
		if had_shoulders {
			drop(skeleton_sides);
			root_skeleton = skeleton.recalculate_root(); 
			skeleton_sides = [&mut skeleton.left, &mut  skeleton.right];
			root_skel_sides = [root_skeleton.left, root_skeleton.right];
		}
		for i in 0..2 {
			let up = match root_skeleton.neck {
				Some(t) => t.up(),
				None => root_skeleton.spine.up()
			};
			let forward = match root_skeleton.neck {
				Some(t) => t.forward(),
				None => root_skeleton.spine.forward()
			};
			// only vaguely shoulder-like
			let virtual_shoulder = root_skel_sides[i].arm.upper.looking_to(forward, up).with_scale(Vec3::ONE);
			// I currently am not confident the following elbow code works correctly
			let final_hand_rel_shoulder = virtual_shoulder.compute_affine().inverse().transform_point3(final_hands[i].translation);
			// these next two lines should account for scale, probably?
			// the coordinate system is unspecified in the paper afaik, I can only pray this works ig
			let model_out = final_hand_rel_shoulder.normalize().mul_add(HEURISTIC_ELBOW_MODEL_WEIGHTS, HEURISTIC_ELBOW_MODEL_BIASES).clamp(Vec3::ZERO, Vec3::INFINITY).dot(Vec3::ONE);
			let elbow_roll = (HEURISTIC_ELBOW_MODEL_OFFSET + model_out).clamp(HEURISTIC_ELBOW_MODEL_CONSTRAINT_MIN, HEURISTIC_ELBOW_MODEL_CONSTRAINT_MAX);
			let mut elbow_vec = Quat::from_axis_angle(final_hand_rel_shoulder, elbow_roll).mul_vec3(Vec3::Y);
			let hand_dist_thresh = HEURISTIC_ELBOW_SINGULARITY_RADIAL_THRESHOLD * height_factor;
			let hand_horiz_dist = (final_hand_rel_shoulder*Vec3::new(1.,0.,1.)).length();
			let hand_dist_scaled = hand_horiz_dist / hand_dist_thresh;
			if hand_dist_scaled < 1. {
				elbow_vec = HEURISTIC_ELBOW_SINGULARITY_VECTOR.lerp(elbow_vec, hand_dist_scaled)
			}
			if final_hand_rel_shoulder.z > HEURISTIC_ELBOW_SINGULARITY_FORWARD_THRESHOLD_MIN && final_hand_rel_shoulder.z < HEURISTIC_ELBOW_SINGULARITY_FORWARD_THRESHOLD_MAX {
				let a = HEURISTIC_ELBOW_SINGULARITY_FORWARD_THRESHOLD_MIN;
				let b = (HEURISTIC_ELBOW_SINGULARITY_FORWARD_THRESHOLD_MAX - a)/2.;
				let lerp_amount = (final_hand_rel_shoulder.z - a + b)/b; // should be 0 at the center, and 1 at the edges
				elbow_vec = HEURISTIC_ELBOW_SINGULARITY_VECTOR.lerp(elbow_vec, lerp_amount);
			}
			let hand_angle_in_arm = virtual_shoulder.looking_to(final_hand_rel_shoulder, elbow_vec).rotation.inverse()*final_hands[i].rotation;
			let hand_angle_euler = hand_angle_in_arm.to_euler(EulerRot::YXZ);
			let hand_yaw = hand_angle_euler.0;
			let hand_roll = hand_angle_euler.2;
			let hand_yaw_thresh_over = hand_yaw - HEURISTIC_ELBOW_WRIST_YAW_THRESHOLD_UPPER;
			let hand_yaw_thresh_under = hand_yaw - HEURISTIC_ELBOW_WRIST_YAW_THRESHOLD_LOWER;
			let hand_roll_thresh_over = hand_roll - HEURISTIC_ELBOW_WRIST_ROLL_THRESHOLD_UPPER;
			let hand_roll_thresh_under = hand_roll - HEURISTIC_ELBOW_WRIST_ROLL_THRESHOLD_LOWER;
			let mut hand_roll_correction: f32 = 0.;
			if hand_yaw_thresh_over > 0. {
				hand_roll_correction += hand_yaw_thresh_over.powi(2)*HEURISTIC_ELBOW_WRIST_YAW_SCALING_CONSTANT_UPPER;
			}
			if hand_yaw_thresh_under < 0. {
				hand_roll_correction += hand_yaw_thresh_under.powi(2)*HEURISTIC_ELBOW_WRIST_YAW_SCALING_CONSTANT_LOWER;
			}
			if hand_roll_thresh_over > 0. {
				hand_roll_correction += hand_roll_thresh_over.powi(2)*HEURISTIC_ELBOW_WRIST_ROLL_SCALING_CONSTANT_UPPER;
			}
			if hand_roll_thresh_under < 0. {
				hand_roll_correction += hand_roll_thresh_under.powi(2)*HEURISTIC_ELBOW_WRIST_ROLL_SCALING_CONSTANT_LOWER;
			}
			elbow_vec = Quat::from_axis_angle(final_hand_rel_shoulder, hand_roll_correction).mul_vec3(elbow_vec);
			// accounting for scaling is fucking annoying
			let upper_arm_length = (root_skel_sides[i].arm.lower.translation - root_skel_sides[i].arm.upper.translation).length_squared();
			let forearm_length = (root_skel_sides[i].hand.translation - root_skel_sides[i].arm.lower.translation).length_squared();
			let shoulder_hand_length = final_hand_rel_shoulder.length_squared();
			// bleigh, this is more annoying than expected
			let hand_upper_arm_angle = cos_law(upper_arm_length, shoulder_hand_length, forearm_length);
			let hand_upper_arm_rot = Quat::from_axis_angle(elbow_vec.cross(final_hand_rel_shoulder), hand_upper_arm_angle);
			// ok, this is what I need, but now it's in the virtual shoulder's coordinate system, which... hmm
			let new_elbow_pos = hand_upper_arm_rot.mul_vec3(final_hand_rel_shoulder).normalize()*upper_arm_length;
			// this feels like a really stupid way of implementing this, but it's also what I can think of immediately so :shrug:
			let curr_elbow_rel_v_shoulder = GlobalTransform::from(root_skel_sides[i].arm.lower).reparented_to(&GlobalTransform::from(virtual_shoulder)).translation;
			let elbow_corrective_rot = Quat::from_rotation_arc(curr_elbow_rel_v_shoulder, new_elbow_pos);
			skeleton_sides[i].arm.upper.rotate(elbow_corrective_rot);
			let updated_root_upper_arm = root_skel_sides[i].arm.upper.with_rotation(root_skel_sides[i].arm.upper.rotation * elbow_corrective_rot);
			let curr_root_lower_arm = skeleton_sides[i].arm.lower.mul_transform(updated_root_upper_arm);
			let target_hand_rel_elbow = GlobalTransform::from(final_hands[i]).reparented_to(&GlobalTransform::from(curr_root_lower_arm));
			let hand_corrective_rot = Quat::from_rotation_arc(skeleton_sides[i].hand.translation, target_hand_rel_elbow.translation);
			skeleton_sides[i].arm.lower.rotate(hand_corrective_rot);
			let updated_root_lower_arm = curr_root_lower_arm.with_rotation(root_skel_sides[i].arm.lower.rotation * hand_corrective_rot);
			let curr_root_hand = skeleton_sides[i].hand.mul_transform(updated_root_lower_arm);
			skeleton_sides[i].hand.rotate(curr_root_hand.rotation.conjugate() * final_hands[i].rotation)
			// WOOOO!!!! OH YEAH BABY THAT'S EVERYTHING!!!!
		}
		drop(skeleton_sides);
		// write the changes to skeleton back to the transforms
		let skeleton_array = skeleton.arrayify();
		for (ent, skel) in skeleton_comp.entities.iter().zip(skeleton_array.iter()) {
			let entity = match ent {Some(e) => e, None => continue};
			let bone = skel.ok_or(eyre::Report::msg("wtf how, Skeleton modified in ways it shouldn't be"))?;
			*transforms.get_mut(*entity)?.0 = bone;
		}
		Ok(())
	};
	let _ = func();

	
}

fn setup_ik(
	mut commands: Commands,
	transforms: Query<(&Transform, &GlobalTransform)>,
	_meshes: ResMut<Assets<Mesh>>,
	_materials: ResMut<Assets<StandardMaterial>>,
	added_query: Query<(Entity, &AvatarSetup)>,
	children: Query<&Children>,
	names: Query<&Name>,
) {
	let skeleton_map: HashMap<&str, usize> = HashMap::from([
		("J_Bip_C_Head", HEAD),
		("J_Bip_C_Hips", HIPS),
		("J_Bip_C_Spine", SPINE),
		("J_Bip_C_Chest", CHEST),
		("J_Bip_C_UpperChest", UPPER_CHEST),
		("J_Bip_C_Neck", NECK),
		("J_Bip_L_Shoulder", LEFT_SHOULDER),
		("J_Adj_L_FaceEye", LEFT_EYE),
		("J_Bip_L_Hand", LEFT_HAND),
		("J_Bip_L_Foot", LEFT_FOOT),
		("J_Bip_L_UpperLeg", LEFT_LEG_UPPER),
		("J_Bip_L_LowerLeg", LEFT_LEG_LOWER),
		("J_Bip_L_UpperArm", LEFT_ARM_UPPER),
		("J_Bip_L_Lowerarm", LEFT_ARM_LOWER),
		("J_Bip_R_Shoulder", RIGHT_SHOULDER),
		("J_Adj_R_FaceEye", RIGHT_EYE),
		("J_Bip_R_Hand", RIGHT_HAND),
		("J_Bip_R_Foot", RIGHT_FOOT),
		("J_Bip_R_UpperLeg", RIGHT_LEG_UPPER),
		("J_Bip_R_LowerLeg", RIGHT_LEG_LOWER),
		("J_Bip_R_Upperarm", RIGHT_ARM_UPPER),
		("J_Bip_R_Lowerarm", RIGHT_LEG_LOWER)
	]);
	for (entity, _thing) in added_query.iter() {
		let mut entities: [Option<Entity>; SKELETON_ARR_LEN] = [None; SKELETON_ARR_LEN];
		let mut root_opt: Option<Entity> = None; 
		// Go through all entities and figure out which ones are important bones
		for child in children.iter_descendants(entity) {
			if let Ok(name) = names.get(child) {
				match skeleton_map.get(name.as_str()) {
					Some(index) => {
						entities[*index] = Some(child);
						commands.entity(child).insert(Bone);
					},
					None => if name.as_str() == "Root" {root_opt = Some(child)}
				}
			}
		};
		let root = match root_opt {
			Some(e) => e,
			None => return,
		};
		// Conversion to skeleton, to check if everthing is initialized, figure out the height of the avatar, and get default offsets
		// If this causes performance problems, looking up the transform of the entire skeleton is unnecessary; this is only necessary
		// for the root and the head.
		let mut skeleton_transform_array: [Option<Transform>; SKELETON_ARR_LEN] = Default::default();
		let mut skeleton_root_array: [Option<Transform>; SKELETON_ARR_LEN] = Default::default();
		let mut func = || -> color_eyre::Result<()> {
			let root_t = transforms.get(root)?.1;
			for i in 0..SKELETON_ARR_LEN {
				let entity = match entities[i] {Some(e) => e, None => continue};
				skeleton_transform_array[i] = Some(*transforms.get(entity)?.0);
				skeleton_root_array[i] = Some(transforms.get(entity)?.1.reparented_to(root_t));
			};
			Ok(())
		};
		let _ = func();
		let skel = match Skeleton::new(skeleton_transform_array)
		{
			Ok(skel) => skel,
			Err(_) => return,
		};
		let root_skel = match Skeleton::new(skeleton_root_array)
		{
			Ok(skel) => skel,
			Err(_) => return,
		};
		let skeleton = SkeletonComponent {entities: entities, height: root_skel.head.translation.y*HEAD_HEIGHT_FACTOR, defaults: skel};
		if skeleton.height <= 0. {return};
		commands.entity(entity).remove::<AvatarSetup>();
		commands.entity(root).insert(skeleton);
	}
}