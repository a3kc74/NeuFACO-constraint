import matplotlib.pyplot as plt
from multiprocessing import Queue as MPQueue


class VrptwAcoFigure:
    def __init__(self, nodes: list, path_queue: MPQueue):
        """
        Matplotlib drawing must run on the main thread; path search should run in a worker thread.
        When the worker finds a new path, it places it in path_queue and the figure updates automatically.
        Paths in the queue are stored as PathMessage objects.
        Nodes are stored as Node objects; Node.x and Node.y provide the coordinates.

        :param nodes: list of nodes, including the depot.
        :param path_queue: queue of paths produced by the worker thread; each path contains node ids.
        """

        self.nodes = nodes
        self.figure = plt.figure(figsize=(10, 10))
        self.figure_ax = self.figure.add_subplot(1, 1, 1)
        self.path_queue = path_queue
        self._depot_color = 'k'
        self._customer_color = 'steelblue'
        self._line_color = 'darksalmon'

    def _draw_point(self):
        # Draw the depot.
        self.figure_ax.scatter([self.nodes[0].x], [self.nodes[0].y], c=self._depot_color, label='depot', s=40)

        # Draw customers.
        self.figure_ax.scatter(list(node.x for node in self.nodes[1:]),
                               list(node.y for node in self.nodes[1:]), c=self._customer_color, label='customer', s=20)
        plt.pause(0.5)

    def run(self):
        # Draw all nodes first.
        self._draw_point()
        self.figure.show()

        # Read a new path from the queue and draw it.
        while True:
            if not self.path_queue.empty():
                # Keep only the newest path in the queue and discard older paths.
                info = self.path_queue.get()
                while not self.path_queue.empty():
                    info = self.path_queue.get()

                path, distance, used_vehicle_num = info.get_path_info()
                if path is None:
                    print('[draw figure]: exit')
                    break

                # Record lines to remove first; do not remove them during iteration.
                # Otherwise self.figure_ax.lines changes while iterating and some lines may not be removed.
                remove_obj = []
                for line in self.figure_ax.lines:
                    if line._label == 'line':
                        remove_obj.append(line)

                for line in remove_obj:
                    self.figure_ax.lines.remove(line)
                remove_obj.clear()

                # Redraw route lines.
                self.figure_ax.set_title('travel distance: %0.2f, number of vehicles: %d ' % (distance, used_vehicle_num))
                self._draw_line(path)
            plt.pause(1)

    def _draw_line(self, path):
        # Draw route segments according to node indices in path.
        for i in range(1, len(path)):
            x_list = [self.nodes[path[i - 1]].x, self.nodes[path[i]].x]
            y_list = [self.nodes[path[i - 1]].y, self.nodes[path[i]].y]
            self.figure_ax.plot(x_list, y_list, color=self._line_color, linewidth=1.5, label='line')
            plt.pause(0.2)
