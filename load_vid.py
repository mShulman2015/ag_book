import cv2
from vid_transform import Transformer

# video input specs
read_from_webcam = True
# input webcam index
input_cam_index = 0
# from file
input_video_path = './videos/input/'
input_file = 'input.mp4'

# video output specs
output_video_path = './videos/output/'
clean_file_name = 'clean.mp4'
final_file_name = 'final.mp4'

# transformer input specs
transformer_specification_file = './booklet_page_identifier.json'

# setup input
if read_from_webcam:
    cap = cv2.VideoCapture(input_cam_index)
    wait_time = 1
else:
    cap = cv2.VideoCapture(input_video_path + input_file)
    wait_time = int(cap.get(cv2.CAP_PROP_FPS))
# input video statistics
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('video input size: {}'.format((width, height)))

# setup output
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
clean_writer = cv2.VideoWriter(output_video_path + clean_file_name, fourcc, 20.0, (width, height))
final_writer = cv2.VideoWriter(output_video_path + final_file_name, fourcc, 20.0, (width, height))

# setup transformer
v_tf = Transformer(transformer_specification_file)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display original
    # cv2.imshow(clean_file_name, frame)
    # clean_writer.write(frame)

    # compute the location of the pages we're looking for in the the frame
    page_location_info = v_tf.compute_page_location_info(gray)
    final_frame = v_tf.compute_final_frame(gray, page_location_info) # TODO: replace gray -> frame when done

    # Display final
    cv2.imshow(final_file_name, final_frame)
    # final_writer.write(final_frame)

    # keep going untill 'q' key is pressed
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# When everything done, release everything
cap.release()
clean_writer.release()
final_writer.release()
cv2.destroyAllWindows()
