use std::f32::consts::PI;

use bevy::core_pipeline::clear_color::ClearColorConfig;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::math::vec3;
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



const ASSET_FOLDER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/");
// Much of the IK Algorithm is sourced from [1]: doi:10.1145/3281505.3281529
// units in meters and radians
const DEFAULT_HEAD_HEIGHT: f32 = 1.69; // slightly more than the height of the average American (men and women)
const DEFAULT_NECK_LENGTH: f32 = 0.13; // from [1], very close to what SteroKitTest/Tools/AvatarSeleton.cs uses
const DEFAULT_SHOULDER_WIDTH: f32 = 0.31; // from [1], very close to what SteroKitTest/Tools/AvatarSeleton.cs uses
const DEFAULT_SHOULDER_OFFSET: f32 = DEFAULT_SHOULDER_WIDTH/2.;
const DEFAULT_UPPER_ARM_LEGNTH: f32 = (DEFAULT_HEAD_HEIGHT - DEFAULT_SHOULDER_WIDTH)/4.;
const DEFAULT_FOREARM_LENGTH: f32 = DEFAULT_UPPER_ARM_LEGNTH;
const USE_MODEL_PROPORTIONS: bool = true; // if true, overwrites all other default sizes except HEAD_HEIGHT with whatever the avatar is using.
const DEFAULT_NECK_ROTATE_CONSTRAINT: f32 = 0.45 * PI; // 81 degrees (about what my head can do)
const HEURISTIC_CHEST_PITCH_HEIGHT_CONSTANT: f32 = 0.7517 * PI; // ~135.3 degrees, from [1]
const HEURISTIC_CHEST_PITCH_HEAD_CONSTANT: f32 = 0.333; // from [1]
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
const HEURISTIC_ELBOW_WRIST_YAW_SCALING_CONSTANT_LOWER: f32 = -0.75 * PI; // 135 degrees, from [1].
const HEURISTIC_ELBOW_WRIST_YAW_SCALING_CONSTANT_UPPER: f32 = 0.75 * PI; // 135 degrees, from [1]
const HEURISTIC_ELBOW_WRIST_ROLL_THRESHOLD_LOWER: f32 = 0.; // 0 degrees, from [1].
const HEURISTIC_ELBOW_WRIST_ROLL_THRESHOLD_UPPER: f32 = 0.5 * PI; // from [1]
const HEURISTIC_ELBOW_WRIST_ROLL_SCALING_CONSTANT_LOWER: f32 = -0.3 * PI; // ...600^-1 degrees? aka 54 degrees, from [1]. I'm not convinced the multiplicative inverse of a rotation is a sensible concept, I hope I did the math correctly.
const HEURISTIC_ELBOW_WRIST_ROLL_SCALING_CONSTANT_UPPER: f32 = 0.6 * PI; // ...300^-1 degrees? aka 108 degrees, from [1]


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
			(update_ik, setup_ik, true_head_sync),
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
pub struct Skeleton {
	head: Option<Entity>,
	hips: Option<Entity>,
	spine: Option<Entity>,
	chest: Option<Entity>,
	upper_chest: Option<Entity>,
	neck: Option<Entity>,
	left: SkeletonSide,
	right: SkeletonSide
}
struct SkeletonSide {
	shoulder: Option<Entity>,
	eye: Option<Entity>,
	leg: Leg,
	arm: Arm,
	foot: Option<Entity>,
	hand: Option<Entity>,
}
struct Leg {
	upper: Option<Entity>,
	lower: Option<Entity>
}
struct Arm {
	upper: Option<Entity>,
	lower: Option<Entity>
}

impl Skeleton {
	pub fn new() -> Self {
		Self {
			head: None,
			hips: None,
			spine: None,
			chest: None,
			upper_chest: None,
			neck: None,
			left: SkeletonSide {
				shoulder: None,
				eye: None,
				leg: Leg {
					upper: None,
					lower: None
				},
				arm: Arm {
					upper: None,
					lower: None
				},
				foot: None,
				hand: None
			},
			right: SkeletonSide {
				shoulder: None,
				eye: None,
				leg: Leg {
					upper: None,
					lower: None
				},
				arm: Arm {
					upper: None,
					lower: None
				},
				foot: None,
				hand: None
			}
		}
	}
}


fn set_head(skeleton: &mut Skeleton, head: Entity) {skeleton.head = Some(head)}
fn set_hips(skeleton: &mut Skeleton, hips: Entity) {skeleton.hips = Some(hips)}
fn set_spine(skeleton: &mut Skeleton, spine: Entity) {skeleton.spine = Some(spine)}
fn set_chest(skeleton: &mut Skeleton, chest: Entity) {skeleton.chest = Some(chest)}
fn set_upper_chest(skeleton: &mut Skeleton, upper_chest: Entity) {skeleton.upper_chest = Some(upper_chest)}
fn set_neck(skeleton: &mut Skeleton, neck: Entity) {skeleton.neck = Some(neck)}
fn set_left_shoulder(skeleton: &mut Skeleton, left_shoulder: Entity) {skeleton.left.shoulder = Some(left_shoulder)}
fn set_right_shoulder(skeleton: &mut Skeleton, right_shoulder: Entity) {skeleton.right.shoulder = Some(right_shoulder)}
fn set_left_eye(skeleton: &mut Skeleton, left_eye: Entity) {skeleton.left.eye = Some(left_eye)}
fn set_right_eye(skeleton: &mut Skeleton, right_eye: Entity) {skeleton.right.eye = Some(right_eye)}
fn set_left_hand(skeleton: &mut Skeleton, left_hand: Entity) {skeleton.left.hand = Some(left_hand)}
fn set_right_hand(skeleton: &mut Skeleton, right_hand: Entity) {skeleton.right.hand = Some(right_hand)}
fn set_left_foot(skeleton: &mut Skeleton, left_foot: Entity) {skeleton.left.foot = Some(left_foot)}
fn set_right_foot(skeleton: &mut Skeleton, right_foot: Entity) {skeleton.right.foot = Some(right_foot)}
fn set_left_leg_upper(skeleton: &mut Skeleton, left_leg_upper: Entity) {skeleton.left.leg.upper = Some(left_leg_upper)}
fn set_right_leg_upper(skeleton: &mut Skeleton, right_leg_upper: Entity) {skeleton.right.leg.upper = Some(right_leg_upper)}
fn set_left_leg_lower(skeleton: &mut Skeleton, left_leg_lower: Entity) {skeleton.left.leg.lower = Some(left_leg_lower)}
fn set_right_leg_lower(skeleton: &mut Skeleton, right_leg_lower: Entity) {skeleton.right.leg.lower = Some(right_leg_lower)}
fn set_left_arm_upper(skeleton: &mut Skeleton, left_arm_upper: Entity) {skeleton.left.arm.upper = Some(left_arm_upper)}
fn set_right_arm_upper(skeleton: &mut Skeleton, right_arm_upper: Entity) {skeleton.right.arm.upper = Some(right_arm_upper)}
fn set_left_arm_lower(skeleton: &mut Skeleton, left_arm_lower: Entity) {skeleton.left.arm.lower = Some(left_arm_lower)}
fn set_right_arm_lower(skeleton: &mut Skeleton, right_arm_lower: Entity) {skeleton.right.arm.lower = Some(right_arm_lower)}


fn true_head_sync(
	mut head_query: Query<(&mut Transform, &TrueHead)>,
	frame_state: Res<XrFrameState>,
	xr_input: Res<XrInput>,
) {
	let mut func = || -> color_eyre::Result<()> {
		let frame_state = *frame_state.lock().unwrap();
		let a = xr_input
			.head
			.relate(&xr_input.stage, frame_state.predicted_display_time)?;
		for (mut head, _) in head_query.iter_mut() {
			head.rotation = a.0.pose.orientation.to_quat();
		}
		Ok(())
	};
	let _ = func();
}


fn update_ik(
	skeleton_query: Query<&Skeleton>,
	mut transforms: Query<(&mut Transform, With<Bone>)>,
	oculus_controller: Res<OculusController>,
	frame_state: Res<XrFrameState>,
	xr_input: Res<XrInput>,
) {
	let mut headset: Option<Transform> = None;
	let mut left_controller: Option<Transform> = None;
	let mut right_controller: Option<Transform> = None;
	let mut func = || -> color_eyre::Result<()> {
		let frame_state = *frame_state.lock().unwrap();
		let headset_input = xr_input
			.head
			.relate(&xr_input.stage, frame_state.predicted_display_time)?;
		let right_controller_input = oculus_controller
			.grip_space
			.right
			.relate(&xr_input.stage, frame_state.predicted_display_time)?;
		let left_controller_input = oculus_controller
			.grip_space
			.left
			.relate(&xr_input.stage, frame_state.predicted_display_time)?;
		headset = Some(Transform {
			translation: headset_input.0.pose.position.to_vec3(),
			rotation: headset_input.0.pose.orientation.to_quat(),
			scale: Default::default(),
		});
		left_controller = Some(Transform {
			translation: left_controller_input.0.pose.position.to_vec3(),
			rotation: left_controller_input.0.pose.orientation.to_quat(),
			scale: Default::default(),
		});
		right_controller = Some(Transform {
			translation: right_controller_input.0.pose.position.to_vec3(),
			rotation: right_controller_input.0.pose.orientation.to_quat(),
			scale: Default::default(),
		});
		Ok(())
	};
	let _ = func();
	let skeleton = skeleton_query.single();
	let head_ent = match skeleton.head {Some(e) => e, None => return};
	let head = match transforms.get_mut(head_ent) {Ok(t) => t, Err(_) => return};
	
}

fn setup_ik(
	mut commands: Commands,
	_meshes: ResMut<Assets<Mesh>>,
	_materials: ResMut<Assets<StandardMaterial>>,
	added_query: Query<(Entity, &AvatarSetup)>,
	children: Query<&Children>,
	names: Query<&Name>,
) {
	let skeleton_map: HashMap<&str, fn(&mut Skeleton, Entity)> = HashMap::from([
		("J_Bip_C_Head", set_head as fn(&mut Skeleton, Entity)),
		("J_Bip_C_Hips", set_hips as fn(&mut Skeleton, Entity)),
		("J_Bip_C_Spine", set_spine as fn(&mut Skeleton, Entity)),
		("J_Bip_C_Chest", set_chest as fn(&mut Skeleton, Entity)),
		("J_Bip_C_UpperChest", set_upper_chest as fn(&mut Skeleton, Entity)),
		("J_Bip_C_Neck",set_neck as fn(&mut Skeleton, Entity)),
		("J_Bip_L_Shoulder",set_left_shoulder as fn(&mut Skeleton, Entity)),
		("J_Bip_R_Shoulder",set_right_shoulder as fn(&mut Skeleton, Entity)),
		("J_Adj_L_FaceEye",set_left_eye as fn(&mut Skeleton, Entity)),
		("J_Adj_R_FaceEye",set_right_eye as fn(&mut Skeleton, Entity)),
		("J_Bip_L_Hand",set_left_hand as fn(&mut Skeleton, Entity)),
		("J_Bip_R_Hand",set_right_hand as fn(&mut Skeleton, Entity)),
		("J_Bip_L_Foot",set_left_foot as fn(&mut Skeleton, Entity)),
		("J_Bip_R_Foot",set_right_foot as fn(&mut Skeleton, Entity)),
		("J_Bip_L_UpperLeg",set_left_leg_upper as fn(&mut Skeleton, Entity)),
		("J_Bip_R_UpperLeg",set_right_leg_upper as fn(&mut Skeleton, Entity)),
		("J_Bip_L_LowerLeg",set_left_leg_lower as fn(&mut Skeleton, Entity)),
		("J_Bip_R_LowerLeg",set_right_leg_lower as fn(&mut Skeleton, Entity)),
		("J_Bip_L_UpperArm",set_left_arm_upper as fn(&mut Skeleton, Entity)),
		("J_Bip_R_Upperarm",set_right_arm_upper as fn(&mut Skeleton, Entity)),
		("J_Bip_L_Lowerarm",set_left_arm_lower as fn(&mut Skeleton, Entity)),
		("J_Bip_R_Lowerarm",set_right_arm_lower as fn(&mut Skeleton, Entity))
	]);
	for (entity, _thing) in added_query.iter() {
		let mut skeleton: Skeleton = Skeleton::new();
		set_head(&mut skeleton, entity);

		// Try to get the entity for the right hand joint.
		for child in children.iter_descendants(entity) {
			if let Ok(name) = names.get(child) {
				match skeleton_map.get(name.as_str()) {
					Some(func) => {
						func(&mut skeleton, child);
						commands.entity(child).insert(Bone);
					},
					None => {}
				}
			}
		}
		let head = match skeleton.head {
			Some(e) => e,
			None => return,
		};
		let hips = match skeleton.hips {
			Some(e) => e,
			None => return,
		};
		commands.entity(entity).remove::<AvatarSetup>();
		commands.entity(hips).insert(skeleton);
		commands.entity(head).insert(TrueHead);
	}
}