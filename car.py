 # A FINIR AVANT 2 SEMAINES !!!!!

import numpy as np
import pygame as pg
import ml.src.function as ml

class Camera:
    def __init__(self, pos, scale, screen_size):
        self.screen_size = screen_size
        self.set_settings(pos, scale)

        
    def set_settings(self, pos=None, scale=None):
        self.pos = pos
        self.scale = scale
        self.Model = ml.vec2( self.screen_size[0]/self.scale[0],
                              -self.screen_size[1]/self.scale[1])


    def projete(self, points):
        """projete une (2,n)||(n,2) matrices sur l'écran"""
        
        return np.array((points-self.pos) * self.Model + [0, self.screen_size[1]], dtype=np.int32)
    
    def draw_seg(self, seg, window):
        pg.draw.aaline(window, (0,0,0), *self.projete(seg))
    
    def draw_car(self, car, window):
        u = ml.vec2(np.linalg.norm(car.scale)/2, 0)
        theta = ml.atan(self.scale[1]/self.scale[0])
        p1 = car.pos + ml.rotate(u, car.angle - theta)
        p2 = car.pos + ml.rotate(u, car.angle + theta)
        p3 = 2*car.pos - p1
        p4 = 2*car.pos - p2
    
        pg.draw.lines(window, (255,0,0), True, self.projete([p1,p2,p3,p4]))
        
    def draw_route(self, route, window):
        pg.draw.lines(window, (0,0,0), True, (self.projete(route.points)))
        pg.draw.lines(window, (0,0,255), False, (self.projete(route.checkpoints)))
        

class Car:
    identity = 0
    @staticmethod
    def get_identity():
        i = Car.identity
        Car.identity += 1
        return i
    
    def __init__(self, pos=ml.vec2(), scale=ml.vec2(1,1)):
        self.pos = pos
        self.scale = scale
        self.angle = 0
        self.v = 1
        self.identity = Car.get_identity()
        self.score = 0
        self.passed_checkpoints = 0
    
    def update(self, dt):
        self.pos += ml.rotate(ml.vec2(self.v, 0)*dt, self.angle)
        
    def get_score(self, route):
        if self.passed_checkpoints<len(route.checkpoints):
            A0M = self.pos-route.checkpoints[self.passed_checkpoints]
            MA1 = route.checkpoints[self.passed_checkpoints+1] - self.pos
            A0A1 = A0M + MA1
            k = np.dot(A0M, MA1) / np.dot(A0A1, A0A1)
        
        min(np.linalg.norm(route.checkpoints-self.pos))

class Seg:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    @property
    def vec(self):
        return np.array([self.a, self.b])
    

    @staticmethod
    def plan_seq(s1, s2):
        if s1.A.x == s1.B.x:
            return ml.signe(s2.A.x - s1.A.x) #### A FINIR!!!!
class Route:
    def __init__(self, width, length, angles):
        self.w = width
        self.l = length
        self.angles = np.radians(angles)
        lpoints = [ml.vec2(0, self.w/2), ml.vec2(self.l, self.w/2)]
        rpoints = [ml.vec2(0, -self.w/2), ml.vec2(self.l,-self.w/2)]

        d = ml.rotate(ml.vec2(1,0), self.angles[0])
        for i, angle in enumerate(self.angles):
            rpoints.append(rpoints[-1]+ d* self.l)
            lpoints.append(lpoints[-1]+ d* self.l)
            
            if i+1 < self.angles.size:
                d = ml.rotate(d, self.angles[i+1])
                if angle>=0:
                    rpoints.append(lpoints[-1]+[[0,1],[-1,0]]@d*self.w)
                else:
                    lpoints.append(rpoints[-1]+[[0, -1], [1, 0]]@d*self.w)
            
            
            """
                
            if angle>=0:
                lpoints.append(lpoints[i]+d * (self.l+(np.sin(angle)*self.w)))
            else:
                rpoints.append(rpoints[i]+d * (self.l+(np.sin(-angle)*self.w)))
            """
       
        rpoints.reverse()
        self.points = lpoints + rpoints



class Labyrinthe:
    def __init__(self, width, length, angles):
        """
        angles appartient {-1, 0, 1}**n
        | -1 -> gauche
        | 0 -> devant
        | 1  -> droite
        """
        self.w = width
        self.l = length
        self.angles = angles
        lpoints = [ml.vec2(0, self.w/2), ml.vec2(self.l, self.w/2)]
        rpoints = [ml.vec2(0, -self.w/2), ml.vec2(self.l,-self.w/2)]
        checkpoints = [ml.vec2(self.l, 0)]

        L = np.transpose(np.array([[0, 1], [-1, 0]]))
        R = -L
        d = ml.vec2(1, 0)
        for i, angle in enumerate(self.angles):
            if angle==-1:
                rpoints.append(rpoints[-1] + d*self.l)
                checkpoints.append((lpoints[-1] + rpoints[-1])/2)
                d = L@d
                rpoints.append(rpoints[-1] + d*self.l)
                checkpoints.append((lpoints[-1] + rpoints[-1])/2)
        
            if angle == 0:
                rpoints.append(rpoints[-1] + d*self.l)
                lpoints.append(lpoints[-1] + d*self.l)
                checkpoints.append((lpoints[-1] + rpoints[-1])/2)
            
            if angle == 1:
                lpoints.append(lpoints[-1] + d*self.l)
                checkpoints.append((lpoints[-1] + rpoints[-1])/2)
                d = R@d
                lpoints.append(lpoints[-1] + d*self.l)
                checkpoints.append((lpoints[-1] + rpoints[-1])/2)

       
        self.rpoints = np.array(rpoints, np.float)
        self.lpoints = np.array(lpoints, np.float)
        self.points = np.array(lpoints + list(reversed(rpoints)), np.float)
        self.checkpoints = np.array(checkpoints, np.float)
    
        
        
        