import copy
import numpy as np
import cv2
import mediapipe as mp 


class finger_tracker:
    def __init__(self, img):
        self.image_height, self.image_width, _ = img.shape
        self.finger_xs = np.zeros([2, 5]) # [hand_side, finger]
        self.pre_finger_ys = np.zeros([2, 5])
        self.finger_ys = np.zeros([2, 5])
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.set_keyboard()
        self.sound_by_key = None
    
    def set_sound(self, sound_by_key):
        self.sound_by_key = sound_by_key

    def set_keyboard(self):
        """define the borders of keyboard"""
        self.key_top = self.image_height // 4
        self.key_bottom = self.image_height//4*3
        self.key_width = self.image_width // 10
        self.key_l_pos = [self.key_width*(2+i) for i in range(9)] # last is the right end of the last key

    def draw_piano(self, image):
        """draw a virtual piano keyboard on the image."""
        # rectangles
        color = (255, 0, 0)
        thickness = 2
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        self.key_name = ["1","2","3","4","5","6","7","8"]

        for i in range(8):
            # draw the keys
            start_point = (self.key_l_pos[i], self.key_top)
            end_point = (self.key_l_pos[i+1], self.key_bottom)        
            text_org = (self.key_l_pos[i]+self.key_width//2, self.image_height//2)

            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            image = cv2.putText(image, self.key_name[i], text_org, font, fontScale, color, thickness, cv2.LINE_AA)

        return image


    def process(self, img):
        """get tip positions and draw finger tracks on the image"""
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = self.hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # draw the keyboard of the piano
        img = self.draw_piano(img)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:               
                self.mpDraw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)
                self.get_hit(img, handLms)
        return img


    def get_hit(self, image, hand_landmarks):
        """
        ['THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP', 'PINKY_TIP']
        """
        hand_side = 0 # left hand
        if hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].x:
            hand_side = 1

        self.pre_finger_ys = copy.deepcopy(self.finger_ys)

        self.finger_xs[hand_side, 0] = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * self.image_width
        self.finger_xs[hand_side, 1] = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * self.image_width
        self.finger_xs[hand_side, 2] = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * self.image_width
        self.finger_xs[hand_side, 3] = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].x * self.image_width
        self.finger_xs[hand_side, 4] = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].x * self.image_width

        self.finger_ys[hand_side, 0] = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * self.image_height
        self.finger_ys[hand_side, 1] = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * self.image_height
        self.finger_ys[hand_side, 2] = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * self.image_height
        self.finger_ys[hand_side, 3] = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y * self.image_height
        self.finger_ys[hand_side, 4] = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y * self.image_height

        threshold = self.image_height//8 # threshold of if the finger hit

        hit_ys = self.finger_ys-self.pre_finger_ys
        hit_xs = self.finger_xs[hit_ys>threshold]
        for hit_x in hit_xs:
            color = (0, 255, 0)
            thickness = 4
            for i in range(len(self.key_l_pos)-1):
                if self.key_l_pos[i]<hit_x and self.key_l_pos[i+1]>hit_x:
                    start_point = (self.key_l_pos[i], self.key_top)
                    end_point = (self.key_l_pos[i+1], self.key_bottom)
                    image = cv2.rectangle(image, start_point, end_point, color, thickness)
                    if self.sound_by_key:
                        self.sound_by_key[self.key_name[i]].play(fade_ms=50)

        return image
                    






if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    ft = finger_tracker(img)

    while True:
        success, img = cap.read()

        if success:
            img = ft.process(img)
        
        cv2.imshow('', img)
        cv2.waitKey(1)