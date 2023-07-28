#image to video modifications
import argparse
import cv2
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    #graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_frame(frame,
                           input_height=299,
                           input_width=299,
                           input_mean=0,
                           input_std=255):
    float_caster = tf.cast(frame, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.compat.v1.Session()
    return sess.run(normalized)

def load_labels(label_file):
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    return [l.rstrip() for l in proto_as_ascii_lines]

if __name__ == "__main__":
    video_file = "D:/Graymatics/TF/tensorflow/tensorflow/examples/label_image/videos/merged.mp4"
    model_file = "D:/Graymatics/TF/inception_v3_2016_08_28_frozen.pb"
    label_file = "D:/Graymatics/TF/imagenet_slim_labels.txt"
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.video:
        video_file = args.video
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    graph = load_graph(model_file)

    cap = cv2.VideoCapture(video_file)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    labels = load_labels(label_file)

    with tf.compat.v1.Session(graph=graph) as sess:
        while(cap.isOpened()):
            ret, frame = cap.read()
            #cv2.imshow('demo',frame)
            if ret == True:
                #cv2.imshow('muffin',frame)
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame', 800, 800)

                frame = cv2.resize(frame, (input_height, input_width))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t = np.expand_dims(frame, axis=0)
                t = t/255.0
    
                results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t})
                results = np.squeeze(results)
                
                # Get the top 5 predictions and their labels
                top_k = results.argsort()[-5:][::-1]
                labels = load_labels(label_file)
                
                # Add labels to the frame to the top 5 predictions
                counter=0
                for i in top_k:
                    # label and score for the current prediction
                    label = labels[i]
                    score = results[i]
                    print(label,score)
                    print("==")
                    x=10
                    y=50*(counter+1)
                    org=(x,y)
                    print(org)
                    if score > 0.8:
                        frame=cv2.putText(frame,label+str(score),org,cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0),1,cv2.LINE_AA)
                
                    else:
                        frame=cv2.putText(frame,label+str(score),org,cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0, 255),1,cv2.LINE_AA)
                    counter+=1          
              
                # frame=cv2.putText(frame,"muffin",(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
                cv2.imshow('frame',frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
                
       
        cap.release()
        cv2.destroyAllWindows()

