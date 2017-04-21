from PIL import Image, ImageDraw
import face_recognition
import numpy
import cv2

def main():
    video_capture = cv2.VideoCapture(0)
    
    # Load a sample picture and learn how to recognize it.
    sivert_image = face_recognition.load_image_file("sivert.jpg")
    sivert_face_encoding = face_recognition.face_encodings(sivert_image)[0]
    
    aksel_image = face_recognition.load_image_file("aksel.jpg")
    aksel_face_encoding = face_recognition.face_encodings(aksel_image)[0]
    
    audun_image = face_recognition.load_image_file("audun.jpg")
    audun_face_encoding = face_recognition.face_encodings(audun_image)[0]
    
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
                match = face_recognition.compare_faces([sivert_face_encoding, aksel_face_encoding, audun_face_encoding], face_encoding)
                name = "Unknown"
    
                if match[0]:
                    name = "Sivert"
                if match[1]:
                    name = "Aksel"
                if match[2]:
                    name = "Audun"
    
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
    
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
        pil_image = Image.fromarray(frame)
        d = ImageDraw.Draw(pil_image, 'RGBA')
        
    
        for i in xrange(len(face_landmarks_list)):
            d.polygon(face_landmarks_list[i]['left_eyebrow'], fill=(68, 54, 39, 128))
            d.polygon(face_landmarks_list[i]['right_eyebrow'], fill=(68, 54, 39, 128))
            d.line(face_landmarks_list[i]['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
            d.line(face_landmarks_list[i]['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
            
            # Gloss the lips
            d.polygon(face_landmarks_list[i]['top_lip'], fill=(150, 0, 0, 128))
            d.polygon(face_landmarks_list[i]['bottom_lip'], fill=(150, 0, 0, 128))
            d.line(face_landmarks_list[i]['top_lip'], fill=(150, 0, 0, 64), width=8)
            d.line(face_landmarks_list[i]['bottom_lip'], fill=(150, 0, 0, 64), width=8)
            
            # Sparkle the eyes
            d.polygon(face_landmarks_list[i]['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks_list[i]['right_eye'], fill=(255, 255, 255, 30))
            
            #apply eyeliner
            d.line(face_landmarks_list[i]['left_eye'] + [face_landmarks_list[i]['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
            d.line(face_landmarks_list[i]['right_eye'] + [face_landmarks_list[i]['right_eye'][0]], fill=(0, 0, 0, 110), width=6)
        
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

main()