import cv2
from mtcnn.mtcnn import MTCNN 
from tracker_center import Tracker
import numpy as np 

def test_webcam(trace=False):
	if trace:
		track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

	detector = MTCNN()
	tracker = Tracker(dist_thresh=30.0, max_frames_to_skip=5, max_trace_length=10, trackIdCount=0)
	cap = cv2.VideoCapture(0)
	time_counter = 0
	frame_counter = 0
	total_counter = 0
	total_frame = 0
	while True:
		_, frame = cap.read()
		result = detector.detect_faces(frame)
		frame_counter += 1
		total_frame += 1
		if result != []:
			result = np.asarray(result)
			box_detected = []
			for person in result:
				bounding_box = person['box']
				box_detected.append(bounding_box)
				keypoints = person['keypoints']

				cv2.rectangle(frame,
					(bounding_box[0], bounding_box[1]),
					(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
					(0,155,255),
					2)

				cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 2)
				cv2.circle(frame, (keypoints['right_eye']), 2, (0, 155, 255), 2)
				# cv2.circle(frame, (keypoints['noise']), 2, (0, 155, 255), 2)
				cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
				cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
			tracker.Update(box_detected, total_frame)
			for i in range(len(tracker.tracks)):
				if len(tracker.tracks[i].trace) >= 2:
					if trace:
						for j in range(len(tracker.tracks[i].trace) - 1):
							# Draw trace line
							x1 = tracker.tracks[i].trace[j][0][0]
							y1 = tracker.tracks[i].trace[j][1][0]
							x2 = tracker.tracks[i].trace[j + 1][0][0]
							y2 = tracker.tracks[i].trace[j + 1][1][0]
							clr = tracker.tracks[i].track_id % 9
							# print(result)
							cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
									track_colors[clr], 2)
					
					if (len(tracker.tracks[i].trace) >= 8) and (not tracker.tracks[i].counted):
						if tracker.tracks[i].ground_truth_box is not None:
							# print("*"*100)
							# print(tracker.tracks[i].ground_truth_box)
							try:
								bbox = tracker.tracks[i].ground_truth_box.reshape((4, 1))
								cv2.rectangle(result, (bbox[0][0], bbox[1][0]), (bbox[2][0], bbox[3][0]),
									color=(255, 0, 255),
									thickness=3)
							except:
								pass

		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cap.destroyAllWindows()

def test_image(image):
	detector = MTCNN()

	image = cv2.imread(image)
	result = detector.detect_faces(image)

	# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
	bounding_box = result[0]['box']
	keypoints = result[0]['keypoints']

	cv2.rectangle(image,
					(bounding_box[0], bounding_box[1]),
					(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
					(0,155,255),
					2)

	cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
	cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
	cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
	cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
	cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

	cv2.imshow("ivan_drawn.jpg", image)
	cv2.waitKey(0)

if __name__ == '__main__':
	# detector = MTCNN()
	test_webcam(trace=False)
	# test_image("ivan.jpg")
