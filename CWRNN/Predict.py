import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


path = os.path.abspath("output/checkpoints/")
print(path)

# Load logs
rop = np.genfromtxt('rop.txt')
wob = np.genfromtxt('wob.txt')
rpm = np.genfromtxt('rpm.txt')
tqa = np.genfromtxt('tqa.txt')
mse = np.genfromtxt('mse.txt')
gr = np.genfromtxt('gr.txt')
rhob = np.genfromtxt('rhob.txt')
nphi = np.genfromtxt('nphi.txt')


portion = .95
train_split = int(portion * rop.shape[0])

p2 = 1
test_split = int((1 - portion) * rop.shape[0] * p2) + train_split

# Split logs
rop = rop[train_split:test_split, :]
wob = wob[train_split:test_split, :]
rpm = rpm[train_split: test_split, :]
tqa = tqa[train_split: test_split, :]
mse = mse[train_split: test_split,:]
gr = gr[train_split: test_split, :]
# nphi = nphi[train_split: test_split, :]
rhob = rhob[train_split: test_split, :]

print(np.shape(rop))

b = 128
rop = rop[:b, :]
wob = wob[:b, :]
rpm = rpm[:b, :]
tqa = tqa[:b, :]
mse = mse[:b, :]
gr = gr[:b, :]
rhob = rhob[:b, :]
# nphi = nphi[:b, :]

newdata = np.stack((gr, rop, rpm,  wob, tqa, mse), axis=2)
num = np.random.choice(np.array(range(b)))
print(np.shape(newdata))
d1 = np.array(list(range(1, rop.shape[1]+1)))

checkpoint_file = tf.train.latest_checkpoint(path)
graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        print(graph.get_operations())
        inputs = graph.get_operation_by_name("inputs").outputs[0]
        # state = graph.get_operation_by_name("clockwork_cell/state/read").outputs[0]
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # state = graph.get_tensor_by_name("clockwork_cell/state/read:0")
        predictions = graph.get_operation_by_name('clockwork_cell/predictions').outputs[0]
        print(predictions)
        # W = graph.get_tensor_by_name('hidden_W')
        # print(W)

        pred = sess.run([predictions], feed_dict={inputs: newdata})
        pred = np.squeeze(pred)

        print(np.shape(pred))
        print(np.shape(rhob))

        plt.interactive(False)
        plt.title("Predicition")
        plt.plot(pred[num, :], d1, Linewidth=2, color='r', label="Predicted Log")
        plt.plot(rhob[num, :], d1, Linewidth=2, color='b', ls='--', label="Actual Log")
        legend = plt.legend(frameon=True)
        # legend.get_frame().set_facecolor('white')
        plt.ylabel('Depth (m)')
        plt.gca().invert_yaxis()
        plt.show()

