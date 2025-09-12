import numpy as np
import habitat
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode

def to_list3(v):
    """Return a flat list[3] for Vector3-like / numpy arrays."""
    try:
        a = np.array(v, dtype=float).reshape(-1)
        if a.size >= 3:
            return [float(a[0]), float(a[1]), float(a[2])]
        return [float(x) for x in a.tolist()]
    except Exception:
        return v

def print_semantic_objects(sim, num=1):
    """
    Print up to `num` SemanticObjects from the current semantic scene.
    Handles both habitat_sim.geo.BBox (center/sizes) and magnum.Range3D (min/max).
    """
    ss = sim.semantic_scene
    assert ss is not None and len(ss.objects) > 0, (
        "No semantic scene or no objects. Use a semantics-enabled dataset/config."
    )

    k = min(num, len(ss.objects))
    print(f"\n=== SemanticObjects (showing {k} of {len(ss.objects)}) ===")
    for i in range(k):
        so = ss.objects[i]
        cat = so.category
        reg = so.region
        aabb = so.aabb
        obb = so.obb

        print(f"\n--- SemanticObject[{i}] ---")
        print(f"id: {so.id}")
        print(f"semantic_id: {so.semantic_id}")

        if cat is not None:
            # SemanticCategory: .name() / .index()
            try:
                print(f"category.name: {cat.name()}  category.index: {cat.index()}")
            except Exception:
                print("category: <unavailable>")

        if reg is not None and getattr(reg, "category", None) is not None:
            try:
                print(f"region.id: {reg.id}")
                print(f"region.category.name: {reg.category.name()}  region.category.index: {reg.category.index()}")
            except Exception:
                pass

        # # ---- AABB handling (BBox vs Range3D) ----
        # if aabb is not None:
        #     if hasattr(aabb, "center") and hasattr(aabb, "sizes"):
        #         # habitat_sim.geo.BBox
        #         c = np.array(aabb.center, dtype=float).reshape(-1)
        #         s = np.array(aabb.sizes, dtype=float).reshape(-1)
        #         mn = c - 0.5 * s
        #         mx = c + 0.5 * s
        #         print("aabb (BBox):")
        #         print(f"  center: {to_list3(c)}  sizes: {to_list3(s)}")
        #         print(f"  min: {to_list3(mn)}  max: {to_list3(mx)}")
        #     elif hasattr(aabb, "min") and hasattr(aabb, "max"):
        #         # magnum.Range3D
        #         print("aabb (Range3D):")
        #         print(f"  min: {to_list3(aabb.min)}  max: {to_list3(aabb.max)}")
        #         try:
        #             print(f"  center: {to_list3(aabb.center())}  size: {to_list3(aabb.size())}")
        #         except Exception:
        #             pass

        # # ---- OBB ----
        # if obb is not None:
        #     print("obb:")
        #     print(f"  center: {to_list3(obb.center)}")
        #     print(f"  sizes: {to_list3(obb.sizes)}  half_extents: {to_list3(obb.half_extents)}")
        #     print(f"  rotation (w,x,y,z): {to_list3(obb.rotation)}")

def print_object_goals(env: habitat.Env, num=1):
    """
    Print up to `num` ObjectGoals from the current episode (ObjectNav).
    """
    ep = env.current_episode
    assert isinstance(ep, ObjectGoalNavEpisode), "Not an ObjectNav episode/config."
    goals = ep.goals or []
    assert len(goals) > 0, "Episode has no ObjectGoal(s)."

    k = min(num, len(goals))
    print(f"\n=== ObjectGoals (showing {k} of {len(goals)}) ===")
    for i in range(k):
        goal = goals[i]
        print(f"\n--- ObjectGoal[{i}] ---")
        print(f"object_id: {goal.object_id}")
        print(f"object_name: {getattr(goal, 'object_name', None)}")
        print(f"object_category: {getattr(goal, 'object_category', None)}")
        print(f"room_id: {getattr(goal, 'room_id', None)}  room_name: {getattr(goal, 'room_name', None)}")
        # print(f"position: {getattr(goal, 'position', None)}  radius: {getattr(goal, 'radius', None)}")

        # vps = getattr(goal, "view_points", []) or []
        # print(f"num view_points: {len(vps)}")
        # if vps:
        #     vp = vps[0]
        #     print(f"  view_point[0].position: {to_list3(getattr(vp, 'position', None))}")
        #     print(f"  view_point[0].radius: {getattr(vp, 'radius', None)}  weight: {getattr(vp, 'weight', None)}")
        #     ast = getattr(vp, "agent_state", None)
        #     if ast is not None:
        #         print(f"  view_point[0].agent_state.position: {to_list3(ast.position)}")
        #         print(f"  view_point[0].agent_state.rotation (w,x,y,z): {to_list3(ast.rotation)}")
