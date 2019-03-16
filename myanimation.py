

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self,skeleton,frame=20):
        # self.stream = self.data_stream()
        self.fig = plt.figure()
        self.stream=skeleton
        self.ax = self.fig.add_subplot(111, projection='3d')
        # plt.xlim((-1, 1))
        # plt.ylim((-1, 1))
        # self.ax.set_zlim(-2, 2)
        # self.ax.view_init( azim=90)
        #self.ax=self.fig.add_subplot(1,1,1)
        self.ani = animation.FuncAnimation(self.fig, self.update,frames=frame,interval=20,init_func = self.setup_plot, blit = True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        frame = self.stream[:, :, 0]
        x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]
        self.graph, = self.ax.plot(x, z,y,linestyle="", marker="o")
        # self.scat=self.ax.scatter(x,y)
        self.title = self.ax.set_title('3D Test')
        return self.graph,

    def update(self, i):
        """Update the scatter plot."""
        print(i)
        frame=self.stream[:,:,i]
        x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]

        self.title.set_text('3D Test, time={}'.format(i))
        # x=np.zeros([20,1])
        # y = np.zeros([20, 1])
        # z = np.zeros([20, 1])
        # self.scat._offsets3d=juggle_axes(x, y,z,'z')

        self.graph.set_data(x, z)
        self.graph.set_3d_properties(y)
        # self.scat.set_offsets(np.random.random((20,2)))
        return self.graph,

    def show(self):
        plt.show()


