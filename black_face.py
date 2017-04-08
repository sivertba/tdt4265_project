import face_recognition
import numpy
import cv2

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]


# Example: convex hull of a 10-by-10 grid.
assert convex_hull([(i//10, i%10) for i in range(100)]) == [(0, 0), (9, 0), (9, 9), (0, 9)]

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

	former_array = []
	t = []
	
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
	#    for (top, right, bottom, left), name in zip(face_locations, face_names):
	#        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
	#        top *= 4
	#        right *= 4
	#        bottom *= 4
	#        left *= 4
	#
	#        # Draw a box around the face
	#        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), 3)
	#
	#        # Draw a label with a name below the face
	#        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 1)
	#        font = cv2.FONT_HERSHEY_DUPLEX
	#        cv2.putText(frame, name, (left + 100, bottom - 10), font, 1.0, (255, 255, 255), 1)
	
	    # Draw on image
	    #dictionary with the follwing keys.
	
	
	
	    face_landmarks_list = face_recognition.face_landmarks(frame)
	    array =[[]]
	    for face_landmarks in face_landmarks_list:
	    	for key in face_landmarks:
	    		for points in face_landmarks[key]:
	    			array[0].append(points)
	   	
	   	array = [convex_hull(array[0])]
	    array = numpy.array(array)

	    if array.size == 0:
	    	cv2.fillPoly(frame, former_array, 0)
	    else:
	    	cv2.fillPoly(frame, array, 0)
	    	former_array = array
	
	    cv2.imshow('Video', frame)
	
	    # Hit 'q' on the keyboard to quit!
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	
	# Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()

main()