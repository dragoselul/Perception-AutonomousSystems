import numpy as np

# Load calibration data
data = np.load('stereo_calib.npz')

print("="*60)
print("STEREO CALIBRATION PARAMETERS")
print("="*60)

print("\n📷 LEFT CAMERA INTRINSICS:")
print("Camera Matrix (ml):")
print(data['ml'])
print("\nDistortion Coefficients (dl):")
print(data['dl'])

print("\n📷 RIGHT CAMERA INTRINSICS:")
print("Camera Matrix (mr):")
print(data['mr'])
print("\nDistortion Coefficients (dr):")
print(data['dr'])

print("\n🔄 STEREO EXTRINSICS:")
print("Rotation Matrix (R) - from left to right camera:")
print(data['R'])
print("\nTranslation Vector (T) - from left to right camera:")
print(data['T'])
print(f"\nBaseline: {np.linalg.norm(data['T']):.2f} units")

print("\n📐 RECTIFICATION MATRICES:")
print("R1 (left camera rectification rotation):")
print(data['R1'])
print("\nR2 (right camera rectification rotation):")
print(data['R2'])

print("\n📊 PROJECTION MATRICES:")
print("P1 (left camera projection):")
print(data['P1'])
print("\nP2 (right camera projection):")
print(data['P2'])

print("\n🎯 DISPARITY-TO-DEPTH MATRIX (Q):")
print(data['Q'])

print("\n" + "="*60)
print("Available keys:", data.files)
print("="*60)

data.close()
