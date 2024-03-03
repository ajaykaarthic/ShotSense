import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# Mapping dictionary to map keypoints from Mediapipe to our Classifier model
lm_dict = {
  0:0 , 1:10, 2:12, 3:14, 4:16, 5:11, 6:13, 7:15, 8:24, 9:26, 10:28, 11:23, 12:25, 13:27, 14:5, 15:2, 16:8, 17:7,
}

def set_pose_parameters():
    mode = False 
    complexity = 1
    smooth_landmarks = True
    enable_segmentation = False
    smooth_segmentation = True
    detectionCon = 0.5
    trackCon = 0.5
    mpPose = mp.solutions.pose
    return mode,complexity,smooth_landmarks,enable_segmentation,smooth_segmentation,detectionCon,trackCon,mpPose

def get_pose (img, results, draw=True):        
        if results.pose_landmarks:
            if draw:
                mpDraw = mp.solutions.drawing_utils
                mpDraw.draw_landmarks(img,results.pose_landmarks,
                                           mpPose.POSE_CONNECTIONS) 
        return img

def get_position(img, results, height, width, draw=True ):
        landmark_list = []
        if results.pose_landmarks:
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                #finding height, width of the image printed
                height, width, c = img.shape
                #Determining the pixels of the landmarks
                landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])
                if draw:
                    cv2.circle(img, (landmark_pixel_x, landmark_pixel_y), 5, (255,0,0), cv2.FILLED)
        return landmark_list    

def get_angle(img, landmark_list, point1, point2, point3, draw=True):   
        #Retrieve landmark coordinates from point identifiers
    
        x1, y1 = landmark_list[point1][1:]
        x2, y2 = landmark_list[point2][1:]
        x3, y3 = landmark_list[point3][1:]
            
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        
        #Handling angle edge cases: Obtuse and negative angles
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle
            
        if draw and (point1 == 11 or point1 == 23) and (point2 == 13 or point2 == 25) and (point3 == 15 or point3 == 27):
            #Drawing lines between the three points
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)

            #Drawing circles at intersection points of lines
            cv2.circle(img, (x1, y1), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (75,0,130), 2)
            cv2.circle(img, (x2, y2), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (75,0,130), 2)
            cv2.circle(img, (x3, y3), 5, (75,0,130), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (75,0,130), 2)
            
            #Show angles between lines
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        return angle

def convert_mediapipe_keypoints_for_model(lm_dict, landmark_list):
    inp_pushup = []
    for index in range(0, 36):
        if index < 18:
            inp_pushup.append(round(landmark_list[lm_dict[index]][1],3))
        else:
            inp_pushup.append(round(landmark_list[lm_dict[index-18]][2],3))
    return inp_pushup

# Setting variables for video feed
def set_video_feed_variables(file_path):
    cap = cv2.VideoCapture(file_path)
    count = 0
    direction = 0
    form = 0
    feedback = "Bad Form."
    frame_queue = deque(maxlen=250)
    return cap,count,direction,form,feedback,frame_queue

def set_percentage_bar_and_text(elbow_angle, knee_angle):
    success_percentage = np.interp(knee_angle, (90, 160), (0, 100))
    progress_bar = np.interp(knee_angle, (90, 160), (380, 30))
    return success_percentage,progress_bar

def set_body_angles_from_keypoints(get_angle, img, landmark_list):
    elbow_angle = get_angle(img, landmark_list, 11, 13, 15)
    shoulder_angle = get_angle(img, landmark_list, 13, 11, 23)
    hip_angle = get_angle(img, landmark_list, 11, 23,25)
    elbow_angle_right = get_angle(img, landmark_list, 12, 14, 16)
    shoulder_angle_right = get_angle(img, landmark_list, 14, 12, 24)
    hip_angle_right = get_angle(img, landmark_list, 12, 24,26)
    knee_angle = get_angle(img, landmark_list, 23, 25, 27)
    return elbow_angle,shoulder_angle,hip_angle,elbow_angle_right,shoulder_angle_right,hip_angle_right,knee_angle

def set_smoothened_workout_name(lm_dict, convert_mediapipe_keypoints_for_model, frame_queue, clf, landmark_list):
    inp_pushup = convert_mediapipe_keypoints_for_model(lm_dict, landmark_list)
    workout_name = clf.predict(inp_pushup)
    frame_queue.append(workout_name)
    workout_name_after_smoothening = max(set(frame_queue), key=frame_queue.count)
    return "Workout Name: " + workout_name_after_smoothening

def run_full_workout_motion(count, direction, form, elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, success_percentage, feedback, workout_name_after_smoothening):
    if workout_name_after_smoothening.strip() == "pushups":
        if form == 1:
            if success_percentage == 0:
                if elbow_angle <= 90 and hip_angle > 160 and elbow_angle_right <= 90 and hip_angle_right > 160:
                    feedback = "Feedback: Go Up"
                    if direction == 0:
                        count += 0.5
                        direction = 1
                else:
                    feedback = "Feedback: Bad Form."
                        
            if success_percentage == 100:
                if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160:
                    feedback = "Feedback: Go Down"
                    if direction == 1:
                        count += 0.5
                        direction = 0
                else:
                    feedback = "Feedback: Bad Form."
        return [feedback, count]
    # For now, else condition handles just squats
    elif workout_name_after_smoothening.strip() == "squats":
        if form == 1:
            if success_percentage == 0:
                if knee_angle < 90:
                    feedback = "Go Up"
                    if direction == 0:
                        count += 0.5
                        direction = 1
                else:
                    feedback = "Feedback: Bad Form."                    
            if success_percentage == 100:
                if knee_angle > 169:
                    feedback = "Feedback: Go Down"
                    if direction == 1:
                        count += 0.5
                        direction = 0
                else:
                    feedback = "Feedback: Bad Form."
            return [feedback, count]
    else:
        return ["Feedback:",0]

def draw_percentage_progress_bar(knee_form, img, success_percentage, progress_bar):
    xd, yd, wd, hd = 10, 175, 50, 200
    
    cv2.rectangle(img, (xd,30), (xd+wd, yd+hd), (0, 255, 0), 3)
    cv2.putText(img, f'{0}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)
    if knee_form == 1:
        cv2.rectangle(img, (xd, int(progress_bar)), (xd+wd, yd+hd), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(success_percentage)}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)

def show_workout_feedback(feedback, img):    
    xf, yf = 85, 70
    cv2.putText(img, feedback, (xf, yf), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0,0,0), 2)

def check_form(elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, form, workout_name_after_smoothening):
    if workout_name_after_smoothening == "pushups":
        if elbow_angle > 160 and shoulder_angle > 40 and hip_angle > 160 and elbow_angle_right > 160 and shoulder_angle_right > 40 and hip_angle_right > 160:
            form = 1
    # For now, else impleements squats condition        
    else:
        if knee_angle > 160:
            form = 1
    return form

def display_workout_stats(count, form, feedback, draw_percentage_progress_bar, show_workout_feedback, img, success_percentage, progress_bar):
    #Draw the progress bar
    draw_percentage_progress_bar(form, img, success_percentage, progress_bar)
        
    #Show the feedback 
    show_workout_feedback(feedback, img)


def main(file_path):
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks,
                                enable_segmentation, smooth_segmentation,
                                detectionCon, trackCon)


    # Setting video feed variables
    cap, count, direction, form, feedback, frame_queue = set_video_feed_variables(file_path)

    #Start video feed and run workout
    knee_form = 0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)
    print(file_path)
    output = cv2.VideoWriter(f"results/{file_path.split('/')[-1].split('.')[0]}.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)
    
    while cap.isOpened():
        #Getting image from camera
        ret, img = cap.read()
        if img is None:
            break
        
        #Getting video dimensions
        width  = cap.get(3)  
        height = cap.get(4)  
        
        #Convert from BGR (used by cv2) to RGB (used by Mediapipe)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        #Get pose and draw landmarks
        img = get_pose(img, results, False)
        
        # Get landmark list from mediapipe
        landmark_list = get_position(img, results, height, width, False)
        
        #If landmarks exist, get the relevant workout body angles and run workout. The points used are identifiers for specific joints
        if len(landmark_list) != 0:
            elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle = set_body_angles_from_keypoints(get_angle, img, landmark_list)

            #Is the form correct at the start?
            success_percentage, progress_bar = set_percentage_bar_and_text(elbow_angle, knee_angle)
            
            #Full workout motion
            if knee_angle < 100 and knee_form == 0:
                knee_form = 1
                print(type(success_percentage), progress_bar)
            if elbow_angle > 45 and elbow_angle < 60:
                feedback = "Feedback: Correct posture for a perfect shot"
            elif elbow_angle > 60 and elbow_angle < 90:
                feedback = "Feedback: Correct posture for a good shot"
            elif elbow_angle > 120:
                feedback = "Feedback: Bend your elbows to make a 45 degree"

            #Display workout stats        
            display_workout_stats(count, knee_form, feedback, draw_percentage_progress_bar, show_workout_feedback, img, success_percentage, progress_bar)
            
            
        # Transparent Overlay
        overlay = img.copy()
        x, y, w, h = 75, 10, 900, 100
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)      
        alpha = 0.75  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)          
        
        output.write(image_new)
        cv2.imshow('SHOT SENSE', image_new)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("/videos/IMG_4290.MOV")
