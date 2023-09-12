use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::math::vec3;
use bevy::prelude::*;
use bevy::transform::components::Transform;
use bevy_openxr::input::XrInput;
use bevy_openxr::resources::XrFrameState;
use bevy_openxr::xr_input::controllers::XrControllerType;
use bevy_openxr::xr_input::oculus_touch::OculusController;
use bevy_openxr::xr_input::{OpenXrInput, QuatConv, Vec3Conv};
use bevy_openxr::DefaultXrPlugins;

const ASSET_FOLDER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/");

fn main() {
	color_eyre::install().unwrap();

	info!("Running `openxr-6dof` skill");
	App::new()
		.add_plugins(DefaultXrPlugins)
		.add_plugins(OpenXrInput::new(XrControllerType::OculusTouch))
		.add_plugins(LogDiagnosticsPlugin::default())
		.add_plugins(FrameTimeDiagnosticsPlugin)
		.add_plugins(bevy_mod_inverse_kinematics::InverseKinematicsPlugin)
		.add_systems(Startup, setup)
		.add_systems(Update, (hands, setup_ik, head_sync, body_sync))
		.run();
}

#[derive(Component)]
pub struct AvatarSetup;

/// set up a simple 3D scene
fn setup(
	mut commands: Commands,
	mut meshes: ResMut<Assets<Mesh>>,
	assets: Res<AssetServer>,
	mut materials: ResMut<Assets<StandardMaterial>>,
) {
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
				Quat::from_euler(EulerRot::XYZ, 0.0, 180.0_f32.to_radians(), 0.0),
			),
			..default()
		},
		AvatarSetup,
		Avatar,
	));
}

#[derive(Component)]
pub enum Hand {
	Left,
	Right,
}

#[derive(Component)]
pub struct Head;
#[derive(Component)]
pub struct Avatar;
#[derive(Component)]
pub struct Hips;

fn head_sync(
	mut head_query: Query<(&mut Transform, &Head)>,
	frame_state: Res<XrFrameState>,
	xr_input: Res<XrInput>,
) {
	let mut func = || -> color_eyre::Result<()> {
		let frame_state = *frame_state.lock().unwrap();
		let a = xr_input
			.head
			.relate(&xr_input.stage, frame_state.predicted_display_time)?;
		for (mut head, _) in head_query.iter_mut() {
			*head = Transform {
				translation: a.0.pose.position.to_vec3(),
				rotation: a.0.pose.orientation.to_quat(),
				scale: Default::default(),
			};
		}
		Ok(())
	};
	let _ = func();
}

fn body_sync(
	frame_state: Res<XrFrameState>,
	xr_input: Res<XrInput>,
	mut avatar: Query<(&mut Transform, &Hips)>,
) {
	let mut func = || -> color_eyre::Result<()> {
		let frame_state = *frame_state.lock().unwrap();
		let a = xr_input
			.head
			.relate(&xr_input.stage, frame_state.predicted_display_time)?;
		for (mut avatar, _) in avatar.iter_mut() {
			let head_pos = Transform {
				translation: a.0.pose.position.to_vec3(),
				rotation: Quat::IDENTITY,
				scale: vec3(1.0, 1.0, 1.0),
			};
			*avatar = head_pos.with_translation(Vec3 {
				x: head_pos.translation.x,
				y: head_pos.translation.y - 0.6,
				z: head_pos.translation.z,
			})
		}
		Ok(())
	};
	let _ = func();
}

fn hands(
	mut gizmos: Gizmos,
	oculus_controller: Res<OculusController>,
	frame_state: Res<XrFrameState>,
	xr_input: Res<XrInput>,
	mut hands: Query<(&mut Transform, &Hand)>,
) {
	let mut func = || -> color_eyre::Result<()> {
		let frame_state = *frame_state.lock().unwrap();

		let right_controller = oculus_controller
			.grip_space
			.right
			.relate(&xr_input.stage, frame_state.predicted_display_time)?;
		let left_controller = oculus_controller
			.grip_space
			.left
			.relate(&xr_input.stage, frame_state.predicted_display_time)?;
		gizmos.rect(
			right_controller.0.pose.position.to_vec3(),
			right_controller.0.pose.orientation.to_quat(),
			Vec2::new(0.05, 0.2),
			Color::YELLOW_GREEN,
		);
		gizmos.rect(
			left_controller.0.pose.position.to_vec3(),
			left_controller.0.pose.orientation.to_quat(),
			Vec2::new(0.05, 0.2),
			Color::YELLOW_GREEN,
		);
		for (mut transform, hand) in hands.iter_mut() {
			match hand {
				Hand::Left => {
					*transform = Transform {
						translation: left_controller.0.pose.position.to_vec3(),
						rotation: left_controller.0.pose.orientation.to_quat(),
						scale: Default::default(),
					}
				}
				Hand::Right => {
					*transform = Transform {
						translation: right_controller.0.pose.position.to_vec3(),
						rotation: right_controller.0.pose.orientation.to_quat(),
						scale: Default::default(),
					}
				}
			}
		}
		Ok(())
	};

	let _ = func();
}

fn setup_ik(
	mut commands: Commands,
	mut meshes: ResMut<Assets<Mesh>>,
	mut materials: ResMut<Assets<StandardMaterial>>,
	added_query: Query<(Entity, &AvatarSetup)>,
	children: Query<&Children>,
	names: Query<&Name>,
) {
	for (entity, _thing) in added_query.iter() {
		let mut right_hand = None;
		let mut left_hand = None;
		let mut head = None;
		let mut hips = None;
		// Try to get the entity for the right hand joint.
		for child in children.iter_descendants(entity) {
			if let Ok(name) = names.get(child) {
				if name.as_str() == "J_Bip_R_Hand" {
					right_hand.replace(child);
				}
				if name.as_str() == "J_Bip_L_Hand" {
					left_hand.replace(child);
				}
				if name.as_str() == "J_Bip_C_Head" {
					head.replace(child);
				}
				if name.as_str() == "J_Bip_C_Hips" {
					hips.replace(child);
				}
			}
		}
		let right_hand = match right_hand {
			Some(e) => e,
			// keep returning until the model fully loads in and we have found the right hand
			// this is massively inefficient.
			None => return,
		};
		let left_hand = match left_hand {
			Some(e) => e,
			None => return,
		};
		let head = match head {
			Some(e) => e,
			None => return,
		};
		let hips = match hips {
			Some(e) => e,
			None => return,
		};
		commands.entity(entity).remove::<AvatarSetup>();

		let pole_target = commands
			.spawn(PbrBundle {
				transform: Transform::from_xyz(-1.0, 0.4, -0.2),
				mesh: meshes.add(Mesh::from(shape::UVSphere {
					radius: 0.05,
					sectors: 7,
					stacks: 7,
				})),
				material: materials.add(StandardMaterial {
					base_color: Color::GREEN,
					..default()
				}),
				..default()
			})
			.id();
		let target_entity1 = commands
			.spawn((TransformBundle::default(), Hand::Right))
			.id();
		let target_entity2 = commands
			.spawn((TransformBundle::default(), Hand::Left))
			.id();
		let target_entity3 = commands.spawn((TransformBundle::default(), Head)).id();
		let hips_entity = commands.spawn((TransformBundle::default(), Hips)).id();
		// Add an IK constraint to the right hand, using the targets that were created earlier.
		commands
			.entity(left_hand)
			.insert(bevy_mod_inverse_kinematics::IkConstraint {
				chain_length: 3,
				iterations: 20,
				target: target_entity1,
				pole_target: Some(pole_target),
				pole_angle: -std::f32::consts::FRAC_PI_2,
				enabled: true,
			});
		commands
			.entity(right_hand)
			.insert(bevy_mod_inverse_kinematics::IkConstraint {
				chain_length: 3,
				iterations: 20,
				target: target_entity2,
				pole_target: Some(pole_target),
				pole_angle: -std::f32::consts::FRAC_PI_2,
				enabled: true,
			});
		commands
			.entity(head)
			.insert(bevy_mod_inverse_kinematics::IkConstraint {
				chain_length: 1,
				iterations: 20,
				target: target_entity3,
				pole_target: Some(pole_target),
				pole_angle: -std::f32::consts::FRAC_PI_2,
				enabled: true,
			});
		commands
			.entity(hips)
			.insert(bevy_mod_inverse_kinematics::IkConstraint {
				chain_length: 1,
				iterations: 20,
				target: hips_entity,
				pole_target: Some(pole_target),
				pole_angle: -std::f32::consts::FRAC_PI_2,
				enabled: true,
			});
	}
}