import cv2, numpy as np, glob, os, sys
SZS, R_BASE, CRIT, SB_FLG = [(11,7),(7,5),(15,5)], 100, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001), cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_EXHAUSTIVE
get_rad = lambda sz: int(R_BASE*(1+0.3*(sz[0]*sz[1])/50*(1.5 if sz[1]>10 else 1)))
def norm(c,s): c=c.reshape(s[1],s[0],2); c=c[::-1] if c[-1,:,1].mean()<c[0,:,1].mean() else c; c=c[:,::-1] if c[:,-1,0].mean()<c[:,0,0].mean() else c; return c.reshape(-1,1,2)
def match(r,t,s): g=t.reshape(s[1],s[0],2); o=[g,g[::-1],g[:,::-1],g[::-1,::-1]]; return o[np.argmin([np.linalg.norm(x.reshape(-1,1,2)-r) for x in o])].reshape(-1,1,2)
def detect(img,x,y,h=None):
    x,y=int(x),int(y)
    for s in ([h] if h else SZS):
        R=get_rad(s); y1,y2,x1,x2=max(0,y-R),min(img.shape[0],y+R),max(0,x-R),min(img.shape[1],x+R); roi=img[y1:y2,x1:x2]
        for im,f,sb in [(roi,SB_FLG,1),(cv2.equalizeHist(cv2.GaussianBlur(roi,(5,5),1)),SB_FLG,1),(roi,11,0)]:
            ret,c=(cv2.findChessboardCornersSB if sb else cv2.findChessboardCorners)(im,s,f)
            if ret: c=cv2.cornerSubPix(roi,c,(11,11),(-1,-1),CRIT) if not sb else c; c[:,0]+=[x1,y1]; c=norm(c,s); return (c,s) if np.linalg.norm(c.mean(0)-[x,y])<R else (None,None)
    return None,None

class Sel:
    def __init__(self,p,t,pts=None): self.img,self.t,self.pts=cv2.imread(p),t,pts or []
    def run(self):
        cv2.namedWindow(self.t,cv2.WINDOW_NORMAL); cv2.setMouseCallback(self.t,lambda e,x,y,f,p:self.add(x,y)if e==1 else None)
        while True:
            v=cv2.resize(self.img,None,fx=0.5,fy=0.5)
            for i,p in enumerate(self.pts): cv2.drawChessboardCorners(v,p['sz'],p['c']*0.5,True); cv2.putText(v,str(i),tuple((p['pos']*0.5).astype(int)),1,2,(0,255,0),2)
            cv2.putText(v,f"{len(self.pts)} boards | [Backspace]Undo [Enter]Done",(20,40),1,1.5,(0,255,255),2); cv2.imshow(self.t,v); k=cv2.waitKey(20)
            if k==13: break
            elif k==8 and self.pts: self.pts.pop(); print(f"Undone. {len(self.pts)} boards remaining.")
        cv2.destroyAllWindows(); return self.pts
    def add(self,x,y): c,s=detect(cv2.cvtColor(self.img,6),x*2,y*2); self.pts.append({'pos':c.mean(0).flatten(),'sz':s,'c':c})if c is not None else None

# --- MAIN ---
def main():
    root = os.path.dirname(os.path.abspath(__file__))
    fl = sorted(glob.glob(os.path.join(root, '../34759_final_project_raw/calib/image_02/data/*.png')))
    fr = sorted(glob.glob(os.path.join(root, '../34759_final_project_raw/calib/image_03/data/*.png')))
    
    # 1. SETUP MASTER ROIs
    roi_path = os.path.join(root, 'calibration_rois.npz')
    if os.path.exists(roi_path) and input("Load ROIs? (y/n): ")=='y':
        d = np.load(roi_path, allow_pickle=True)
        lay_l, lay_r = d['lay_l'].tolist(), d['lay_r'].tolist()
    else:
        lay_l = Sel(fl[0], "L-MASTER").run()
        lay_r = Sel(fr[0], "R-MASTER").run()
        np.savez(roi_path, lay_l=lay_l, lay_r=lay_r)

    # Group fallbacks by size
    groups_l, groups_r = {}, {}
    for b in lay_l: groups_l.setdefault(b['sz'], []).append(b)
    for b in lay_r: groups_r.setdefault(b['sz'], []).append(b)

    # 2. PROCESS FRAMES
    obj_pts, img_l, img_r = [], [], []
    sz = cv2.imread(fl[0]).shape[:2][::-1]

    for idx, (f_l, f_r) in enumerate(zip(fl, fr)):
        print(f"Frame {idx}...", end=" ")
        gl, gr = cv2.cvtColor(cv2.imread(f_l), 6), cv2.cvtColor(cv2.imread(f_r), 6)
        det_l, det_r = {}, {}  # Use dict to track board index
        
        # Create masks to prevent duplicate detections
        mask_l = np.ones_like(gl) * 255
        mask_r = np.ones_like(gr) * 255
        
        # Auto-detect using ALL fallback ROIs for each board in original layout
        for board_idx, (tgt_l, tgt_r) in enumerate(zip(lay_l, lay_r)):
            # Try all fallback ROIs with matching size for left camera
            for fb in groups_l.get(tgt_l['sz'], []):
                c, s = detect(gl & mask_l, fb['pos'][0], fb['pos'][1], tgt_l['sz'])
                if c is not None:
                    det_l[board_idx] = {'c':c, 'sz':s}
                    # Mask out detected region
                    min_pt = c.min(axis=0)[0].astype(int)
                    max_pt = c.max(axis=0)[0].astype(int)
                    cv2.rectangle(mask_l, tuple(min_pt), tuple(max_pt), 0, -1)
                    break
            
            # Try all fallback ROIs with matching size for right camera
            for fb in groups_r.get(tgt_r['sz'], []):
                c, s = detect(gr & mask_r, fb['pos'][0], fb['pos'][1], tgt_r['sz'])
                if c is not None:
                    det_r[board_idx] = {'c':c, 'sz':s}
                    # Mask out detected region
                    min_pt = c.min(axis=0)[0].astype(int)
                    max_pt = c.max(axis=0)[0].astype(int)
                    cv2.rectangle(mask_r, tuple(min_pt), tuple(max_pt), 0, -1)
                    break
        
        # Convert dicts to lists (None for missing boards)
        det_l_list = [det_l.get(i) for i in range(len(lay_l))]
        det_r_list = [det_r.get(i) for i in range(len(lay_r))]

        # Fix missing boards manually
        miss = [i for i, (l, r) in enumerate(zip(det_l_list, det_r_list)) if l is None or r is None]
        if miss:
            print(f"Fixing {len(miss)} boards: {miss}")
            # Add 'pos' field to detected boards for Sel class compatibility
            det_l_with_pos = [{'pos': d['c'].mean(0).flatten(), 'sz': d['sz'], 'c': d['c']} if d else None for d in det_l_list]
            det_r_with_pos = [{'pos': d['c'].mean(0).flatten(), 'sz': d['sz'], 'c': d['c']} if d else None for d in det_r_list]
            
            fix_l = Sel(f_l, f"FIX L {miss}", [d for d in det_l_with_pos if d]).run()
            fix_r = Sel(f_r, f"FIX R {miss}", [d for d in det_r_with_pos if d]).run()
            if len(fix_l) == len(lay_l) and len(fix_r) == len(lay_r):
                det_l_list, det_r_list = fix_l, fix_r
            else: 
                print("Skipping frame (incomplete)"); continue

        # Verify and add to calibration data
        print(f"✓ Detected {len([d for d in det_l_list if d])}/{len(det_l_list)} boards")
        
        # Ask if user wants to verify corners
        verify_corners = input("Verify corners for this frame? (y/n): ").lower() == 'y'
        
        for i, (l, r) in enumerate(zip(det_l_list, det_r_list)):
            if l['sz'] == (r['sz'][1], r['sz'][0]): # Handle 90deg rotation
                r['c'] = r['c'].reshape(r['sz'][1], r['sz'][0], 2).transpose(1,0,2).reshape(-1,1,2)
                r['sz'] = l['sz']
            
            r['c'] = match(l['c'], r['c'], l['sz']) # Match orientation
            
            if verify_corners:
                # Show for verification
                img_l_full, img_r_full = cv2.imread(f_l), cv2.imread(f_r)
                vis = np.hstack([img_l_full, img_r_full])
                cv2.drawChessboardCorners(vis, l['sz'], l['c'], True)
                r_vis = r['c'].copy(); r_vis[:,0,0] += img_l_full.shape[1]
                cv2.drawChessboardCorners(vis, l['sz'], r_vis, True)
                cv2.circle(vis, tuple(l['c'][0,0].astype(int)), 12, (0,255,255), -1)
                cv2.circle(vis, tuple(r_vis[0,0].astype(int)), 12, (0,255,255), -1)
                vis = cv2.resize(vis, None, fx=0.5, fy=0.5)
                cv2.putText(vis, f"Board {i} | [Space]OK [H]FlipH [V]FlipV [B]Both [R]Reject", (20,40), 1, 1.5, (0,255,0), 2)
                cv2.imshow("Verify", vis)
                k = cv2.waitKey(0)
                
                if k == ord('h'): r['c'] = r['c'].reshape(l['sz'][1], l['sz'][0], 2)[::-1].reshape(-1,1,2)
                elif k == ord('v'): r['c'] = r['c'].reshape(l['sz'][1], l['sz'][0], 2)[:,::-1].reshape(-1,1,2)
                elif k == ord('b'): r['c'] = r['c'].reshape(l['sz'][1], l['sz'][0], 2)[::-1,::-1].reshape(-1,1,2)
                elif k == ord('r'): print(f"  ✗ Rejected board {i}"); continue
            
            op = np.zeros((l['sz'][0]*l['sz'][1], 3), np.float32)
            op[:,:2] = np.mgrid[0:l['sz'][0], 0:l['sz'][1]].T.reshape(-1,2)
            obj_pts.append(op); img_l.append(l['c']); img_r.append(r['c'])
        
        if verify_corners:
            cv2.destroyWindow("Verify")

    # FINAL REVIEW - Check all collected board correspondences
    print(f"\n{'='*60}")
    print(f"FINAL REVIEW: {len(obj_pts)} board views collected")
    print("Checking corner alignments...")
    
    bad_boards = []
    for idx, (l_c, r_c) in enumerate(zip(img_l, img_r)):
        # Calculate average distance between corresponding corners
        avg_dist = np.mean(np.linalg.norm(l_c - r_c, axis=2))
        if avg_dist > 50:  # Threshold for bad alignment
            bad_boards.append((idx, avg_dist))
    
    if bad_boards:
        print(f"\n⚠ WARNING: {len(bad_boards)} boards have poor corner alignment:")
        for idx, dist in bad_boards:
            print(f"  Board {idx}: avg distance = {dist:.1f} pixels")
        
        if input("\nReview all boards visually? (y/n): ") == 'y':
            boards_to_remove = []
            for idx, (l_c, r_c) in enumerate(zip(img_l, img_r)):
                frame_idx = idx // len(lay_l)
                board_idx = idx % len(lay_l)
                
                img_l_full = cv2.imread(fl[frame_idx])
                img_r_full = cv2.imread(fr[frame_idx])
                vis = np.hstack([img_l_full, img_r_full])
                
                # Draw corners
                sz_board = (int(np.sqrt(len(l_c))), len(l_c) // int(np.sqrt(len(l_c))))
                cv2.drawChessboardCorners(vis, sz_board, l_c, True)
                r_vis = r_c.copy(); r_vis[:,0,0] += img_l_full.shape[1]
                cv2.drawChessboardCorners(vis, sz_board, r_vis, True)
                
                # Highlight first corner
                cv2.circle(vis, tuple(l_c[0,0].astype(int)), 15, (0,255,255), -1)
                cv2.circle(vis, tuple(r_vis[0,0].astype(int)), 15, (0,255,255), -1)
                
                # Calculate metrics
                avg_dist = np.mean(np.linalg.norm(l_c - r_c, axis=2))
                
                vis = cv2.resize(vis, None, fx=0.5, fy=0.5)
                status = "⚠ BAD" if (idx, avg_dist) in [(i, d) for i, d in bad_boards] else "✓ OK"
                cv2.putText(vis, f"{status} | Board {idx} (Frame {frame_idx}, #{board_idx}) | Dist: {avg_dist:.1f}px", 
                           (20, 40), 1, 1.5, (0,0,255) if status == "⚠ BAD" else (0,255,0), 2)
                cv2.putText(vis, "[Space]Keep [R]Remove [ESC]Stop Review", (20, 70), 1, 1.2, (255,255,255), 2)
                cv2.imshow("Final Review", vis)
                k = cv2.waitKey(0)
                
                if k == ord('r'):
                    boards_to_remove.append(idx)
                    print(f"  Marked board {idx} for removal")
                elif k == 27:  # ESC
                    break
            
            cv2.destroyWindow("Final Review")
            
            # Remove marked boards
            if boards_to_remove:
                print(f"\nRemoving {len(boards_to_remove)} boards...")
                for idx in sorted(boards_to_remove, reverse=True):
                    del obj_pts[idx]
                    del img_l[idx]
                    del img_r[idx]
                print(f"✓ {len(obj_pts)} boards remaining")
    else:
        print("✓ All boards have good corner alignment!")

    # 3. CALIBRATE
    if len(obj_pts)<10 or input(f"\nCalibrate with {len(obj_pts)} views? (y/n):")!='y': sys.exit()
    print("Calibrating..."); flags=cv2.CALIB_USE_INTRINSIC_GUESS+cv2.CALIB_FIX_PRINCIPAL_POINT; mtx=np.array([[sz[0]*1.2,0,sz[0]/2],[0,sz[0]*1.2,sz[1]/2],[0,0,1]],np.float32)
    rl,ml,dl,_,_=cv2.calibrateCamera(obj_pts,img_l,sz,mtx,None,flags=flags)
    rr,mr,dr,_,_=cv2.calibrateCamera(obj_pts,img_r,sz,mtx,None,flags=flags)
    ret,ml,dl,mr,dr,R,T,_,_=cv2.stereoCalibrate(obj_pts,img_l,img_r,ml,dl,mr,dr,sz,flags=cv2.CALIB_FIX_INTRINSIC); print(f"RMS: {ret:.2f}")
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(ml,dl,mr,dr,sz,R,T,alpha=0); np.savez("stereo_calib.npz",ml=ml,dl=dl,mr=mr,dr=dr,R=R,T=T,Q=Q,R1=R1,R2=R2,P1=P1,P2=P2)
    
    # RECTIFY AND SAVE
    os.makedirs("rectified/image_02/data",exist_ok=True); os.makedirs("rectified/image_03/data",exist_ok=True)
    m1,m2=cv2.initUndistortRectifyMap(ml,dl,R1,P1,sz,cv2.CV_16SC2); m3,m4=cv2.initUndistortRectifyMap(mr,dr,R2,P2,sz,cv2.CV_16SC2); print("\nRectifying images...")
    for i,(lp,rp) in enumerate(zip(fl,fr)): il,ir=cv2.imread(lp),cv2.imread(rp); fn=os.path.basename(lp); cv2.imwrite(f"rectified/image_02/data/{fn}",cv2.remap(il,m1,m2,cv2.INTER_LINEAR)); cv2.imwrite(f"rectified/image_03/data/{fn}",cv2.remap(ir,m3,m4,cv2.INTER_LINEAR)); print(f"  {i+1}/{len(fl)}: {fn}")
    print(f"\n✓ Saved rectified images to rectified/")

if __name__=="__main__": 
    main()
    