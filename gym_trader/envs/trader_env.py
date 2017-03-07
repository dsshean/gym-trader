import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import sys
from pandas import read_hdf
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.exporters
from random import shuffle
from PIL import Image
import logging
import numpy as np
from datetime import datetime
import collections
import cv2
from skimage import color
from skimage import io

# try:
#     import hfo_py
# except ImportError as e:
#     raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[trader].)'".format(e))

import logging
logger = logging.getLogger(__name__)

class TraderEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.viewer = None
        self.dims = (300, 300)
        self.observation_space = spaces.Box(low=0, high=300,shape=(self.dims))
        # Action space omits the Tackle/Catch actions, which are useful on defense
        self.action_space = spaces.Discrete(3)
        self.reward = 0
        self.actions = self.action_space
        self.df = read_hdf('uptodate.h5')
        self.df = self.df.loc['2000-1-1':'2014-1-1']
        self.grouped = self.df.groupby(lambda x: x.date).filter(lambda x: len(x) > 389 and len(x) < 391)
        self.grouped = self.grouped.groupby(lambda x: x.date)
        self.dates = self.grouped.groups.keys()
        shuffle(self.dates)
        self.epochs = len(self.dates)  # number of epochs = # of trading days we are training for.
        print(self.epochs)
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow()
        self.p1 = self.win.addPlot()
        self.p1.setXRange(0, 390)
        self.terminal = False
        self.text1 = pg.TextItem(text='LONG ', anchor=(0, 0), border='w', fill=(255, 255, 255, 255))
        self.text2 = pg.TextItem(text='SHORT', anchor=(0, 1), border='w', fill=(255, 255, 255, 255))
        self.text3 = pg.TextItem(text='BUY  ', anchor=(1, 1), border='w', fill=(255, 255, 255, 255))
        self.text4 = pg.TextItem(text='SELL ', anchor=(1, 0), border='w', fill=(255, 255, 255, 255))
        self.lineitem = pg.InfiniteLine()
        self.lineitem.setValue(0)
        self.p1.addItem(self.text1)
        self.p1.addItem(self.text2)
        self.p1.addItem(self.text3)
        self.p1.addItem(self.text4)
        self.p1.addItem(self.lineitem)
        self.curve1 = self.p1.plot()
        self.app.processEvents()
        self.counter = 0
        self.aaaa = pg.exporters.ImageExporter(self.p1)
        self.aaaa.export('temp.png')
        self.state = color.rgb2gray(io.imread('temp.png'))
        self.state = np.array(self.state)
        self.data = []
        self.count = 0
        self.cumrewards = 0.0
        self.testrewards1 = []
        self.testprofits1 = []
        self.testrewards = 0.0
        self.testprofits = 0.0
        self.b = "16:00:00"
        self.dt = datetime.strptime(self.b, "%H:%M:%S")
        self.currenttime = datetime.strptime('9:29:00', "%H:%M:%S")
        self.currentdate = self.dates[0]
        self.dates.pop(0)
        self.times = self.grouped.get_group(self.currentdate).index.tolist()
        self.close = self.grouped.get_group(self.currentdate).values.tolist()
        self.position = 0
        self.bprice = 0
        print("done")
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.data.append(self.close[0][3])
        self.counter = len(self.data)
        self.close.pop(0)
        self.times.pop(0)
        if len(self.data) >0:
            self.text1.setPos(0, max(self.data))
            self.text2.setPos(0, min(self.data))
            self.text3.setPos(390, max(self.data))
            self.text4.setPos(390, min(self.data))
        if self.position == 0:
            self.text1.setText(text='LONG ', color='000000')
            self.text2.setText(text='SHORT', color='000000')
        elif self.position == 1:
            self.text1.setText(text='@', color='FFFFFF')
            self.text2.setText(text='SHORT', color='000000')
        elif self.position == -1:
            self.text1.setText(text='LONG ', color='000000')
            self.text2.setText(text='@', color='FFFFFF')
        self.text3.setText(text='BUY  ', color='000000')
        self.text4.setText(text='SELL ', color='000000')
        self.curve1.setData(self.data)
        #self.state = self.init()
        if action == 0:
            self.text3.setText(text='@', color='FFFFFF')
            if self.position == 1:
                self.reward = 0
                self.reward1 = 0
            elif self.position == 0:
                self.text1.setText(text='@', color='FFFFFF')
                self.position = 1
                self.bprice = self.data[-1]
                self.lineitem.setValue(self.counter)
                self.reward = 0
                self.reward1 = 0
        elif action == 1:
            self.text4.setText(text='@', color='FFFFFF')
            if self.position == -1:
                self.reward = 0
                self.reward1 = 0
            elif self.position == 0:
                self.text2.setText(text='@', color='FFFFFF')
                self.position = -1
                self.bprice = self.data[-1]
                self.lineitem.setValue(self.counter)
                self.reward = 0
                self.reward1 = 0
        elif action == 2:
            self.text4.setText(text='@', color='FFFFFF')
            if self.position == 0:
                self.reward = 0
                self.reward1 = 0
            elif self.position == 1:
                self.position = 0
                self.text1.setText(text='LONG ', color='000000')
                self.lineitem.setValue(0)
                self.reward1 =  self.data[-1] - self.bprice
                if self.reward1 > 0:
                    self.reward = 1
                else:
                    self.reward = -1
            elif self.position == -1:
                self.position = 0
                self.text2.setText(text='SHORT', color='000000')
                self.lineitem.setValue(0)
                self.reward1 =  self.bprice - self.data[-1]
                if self.reward1 > 0:
                    self.reward = 1
                else:
                    self.reward = -1
        else:
            self.reward = 0
            self.reward1 = 0
        if self.times[0].time() == self.dt.time():
            print("Terminal" + str(self.dates[0]))
            self.dates.pop(0)
            self.currentdate = self.dates[0]
            self.times = self.grouped.get_group(self.currentdate).index.tolist()
            self.close = self.grouped.get_group(self.currentdate).values.tolist()
            self.position = 0
            self.bprice = 0
            self.terminal = True
            action = 2
            if self.position == 0:
                self.reward = 0
            elif self.position == 1:
                self.position = 0
                self.lineitem.setValue(0)
                self.reward1 =  self.data[-1] - self.bprice
                if self.reward1 > 0:
                    self.reward = 1
                else:
                    self.reward = -1
            elif self.position == -1:
                self.position = 0
                self.lineitem.setValue(0)
                self.reward1 =  self.bprice - self.data[-1]
                if self.reward1 > 0:
                    self.reward = 1
                else:
                    self.reward = -1
            self.data = []
        self.app.processEvents()
        self.aaaa.export('temp.png')
        self.state = color.rgb2gray(io.imread('temp.png'))
        self.state = np.array(self.state)
        return self.state , self.reward, self.terminal, {}

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        return self.reward

    def getState(self):
        self.app.processEvents()
        self.aaaa.export('temp.png')
        self.state = color.rgb2gray(io.imread('temp.png'))
        self.state = np.array(self.state)
        screen = self.state
        resized = cv2.resize(screen, self.dims)
        return resized

    def _reset(self):
        return self.getState()


ACTION_LOOKUP = {
    0 : "LONG",
    1 : "SHORT",
    2 : "CLOSE",
    3 : "NOOP",
}
