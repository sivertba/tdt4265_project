from PIL import Image, ImageDraw
import face_recognition
import numpy
import cv2

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("sivert.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces([obama_face_encoding], face_encoding)
            name = "Unknown"

            if match[0]:
                name = "Sivert"

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 3)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 10, bottom - 10), font, 1.0, (255, 255, 255), 1)

    # Draw on image
    face_landmarks_list = face_recognition.face_landmarks(frame)
    for face_landmarks in face_landmarks_list:
    	
    	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    	pil_image = Image.fromarray(frame)
    	d = ImageDraw.Draw(pil_image, 'RGBA')

    	d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
    	d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
    	d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
    	d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

    	# Gloss the lips
    	d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
    	d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
    	d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
    	d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

    	# Sparkle the eyes
    	d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
    	d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

    	# Apply some eyeliner
    	d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
    	d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

    	outimg = numpy.array(pil_image)
    	outimg = outimg[:, :, ::-1].copy()

    	# Display the resulting image
    	cv2.imshow('Video', outimg)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
