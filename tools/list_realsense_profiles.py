import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.query_devices():
    print(f"\nDevice: {dev.get_info(rs.camera_info.name)} ({dev.get_info(rs.camera_info.serial_number)})")
    for sensor in dev.query_sensors():
        print(f" Sensor: {sensor.get_info(rs.camera_info.name)}")
        for p in sensor.get_stream_profiles():
            vp = p.as_video_stream_profile()
            if vp.fps() == 60:
                print(f"   60 FPS Profile: {vp.stream_type()} | {vp.format()} | {vp.width()}x{vp.height()}")