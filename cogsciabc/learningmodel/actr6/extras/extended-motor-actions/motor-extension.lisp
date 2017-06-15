;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Author      : Dan Bothell
;;; Copyright   : (c) 2014 Dan Bothell
;;; Availability: Covered by the GNU LGPL, see LGPL.txt
;;; Address     : Department of Psychology 
;;;             : Carnegie Mellon University
;;;             : Pittsburgh, PA 15213-3890
;;;             : db30@andrew.cmu.edu
;;; 
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Filename    : motor-extension.lisp
;;; Version     : 1.0a1
;;; 
;;; Description : Collection of extensions for the motor module which include:
;;;             : separate processor and/or execution stages for each hand, buffers
;;;             : to track the separate hand usage, actions for holding keys,
;;;             : and "press-key like" actions for holding keys and for hitting
;;;             : a key without recoiling to the home row.
;;; 
;;; Bugs        : 
;;;
;;; To do       : 
;;; 
;;; ----- History -----
;;; 2014.04.24 Dan [1.0a1]
;;;             : * Initial creation.  Combining the separate extensions that
;;;             :   were used for modeling Space Fortress, cleaning them up so
;;;             :   that the "regular" motor actions work correctly with them,
;;;             :   and adding the new "typing" actions.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; General Docs:
;;; 
;;; Combining a lot of updates to the motor module together in one place which
;;; should be useable as a replacement for the default module without breaking
;;; any existing models which use motor actions.  
;;; 
;;; The big changes with the update:
;;;
;;; The model now generates actions for pressing and releasing keys which can
;;; be detected by the modeler.  The actions from the existing module (punch,
;;; peck, peck-recall, and press-key) all now result in both a press and release
;;; of the key/button.  The presses still call the user handlers:
;;; rpm-window-key-event-handler and rpm-window-click-event-handler 
;;; and now when a key or button is released these handlers will be called:
;;; rpm-window-key-release-event-handler and rpm-window-click-release-event-handler.
;;; Those new handlers are only called for model actions at this point however
;;; and are not tied into the real devices for detecting user key/button releasing.
;;;
;;; With the addition of keys being able to be held down the previous motor module
;;; actions (the styles punch, peck, peck-recoil, and point-hand and all the 
;;; higher-level actions which use them press-key, click-mouse, hand-to-home,
;;; hand-to-mouse, and point-hand-at-key) have been modified so that if the finger(s)
;;; used to perform the action is currently holding a key down there's an extra 
;;; burst-time cost to execute the release of that key before performing the
;;; specified action. 
;;;
;;; The trace of the motor actions for preparing, initiating, and finishing an
;;; action now indicate which action and hand it corresponds to since there may
;;; be parallel actions on going. (The old module also could have some overlapping
;;; actions which occasionally lead to confusion in the trace, but it didn't 
;;; happen very often).
;;;
;;; The new motor module now has the option of using separate execution and/or 
;;; processor stages for the two hands which allows motor actions on different 
;;; hands to potentially operate in parallel.  There is still only one preparation
;;; stage however.  The default operation of the new module is for two execution
;;; stages and one processor stage, but that can be changed with parameters as
;;; described below.
;;;
;;; There are two new sets of motor actions which can be used.  The first set
;;; provides separate low-level actions for holding and releasing keys.  The other 
;;; set is more high-level actions like press-key which perform touch typing 
;;; operations with holding keys and without always recoiling to the home row.
;;; Details on the new actions are found below.
;;;
;;; There is a new module created, named motor-extension, which adds two new
;;; buffers, manual-left and manual-right.  These can be used for tracking the 
;;; operation of the model's hands separately and also allows for querying 
;;; whether individual fingers are currently busy i.e. holding down a key.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Public API:
;;;
;;; rpm-window-key-release-event-handler (device key)
;;;
;;; This method can be defined on a device by the modeler to detect when the 
;;; model releases a key.  It will be called with the name of the key released
;;; just like the rpm-window-key-event-handler does when a key is pressed.
;;; The call to the rpm-window-key-release-event-handler method will always
;;; be proceeded by a call to the rpm-window-key-event-handler for the key,
;;; however there may not always be a call to rpm-window-key-release-event-handler
;;; for every rpm-window-key-event-handler because the model may be reset or
;;; otherwise stopped or cleared before the release event occurs.
;;;
;;; rpm-window-click-release-event-handler (device position)
;;;
;;; This method can be defined on a device by the modeler to detect when the 
;;; model releases the mouse button.  It is passed the position of the mouse
;;; at the time of the button release.  Like the key-release-event-handler,
;;; there will always be a preceding rpm-window-click-event-handler call, but 
;;; every rpm-window-click-event-handler may not have a corresponding call to
;;; the rpm-window-click-release-event-handler.
;;;
;;; 
;;; New model parameters:
;;;
;;; :key-closure-time (default value .01)
;;; 
;;; The time in seconds between when a key is contacted (the corresponding movement 
;;; action has started pressing it) and when it registers as being hit.  By default
;;; only used for the timing of punch actions, but see the :peck-strike-distance
;;; parameter below for how it may affect peck/peck-recoil actions as well.  Note,
;;; this value has always existed within the device interface, but was not previously
;;; made directly available to the modeler.
;;;
;;; :key-release-time (default value .04)
;;;
;;; The time in seconds between when the action which is releasing a key is started
;;; and when that key actually registers as being released.  Used in the timing of
;;; all the actions which include a releasing of a key.  The assumption is that there
;;; is some distance over which the finger must be moved before the key stops being
;;; pressed.  All of the release actions are based on a burst-time cost, thus this
;;; parameter should be set to a value no greater than the value of the :MOTOR-BURST-TIME
;;; parameter.  
;;;
;;; NOTE: Those two parameters should really be modeled as a distance instead of a time
;;; cost, but since the motor movement code isn't modeled at that fine grain of detail
;;; (exact finger positions in 3-space) a simple cost assumption is used.
;;;
;;; :dual-processor-stages (default value nil)
;;;
;;; A boolean indicating whether the motor module should use separate processor
;;; stages for each hand.
;;;
;;; :dual-execution-stages (default value nil)
;;;
;;; A boolean indicating whether the motor module should use separate execution
;;; stages for each hand.
;;;
;;; :peck-strike-distance (default 0.0)
;;; 
;;; The distance from the center of a key at which a peck action is considered
;;; to make contact with the key and start depressing it.  The value can be
;;; from 0 to .5 (the width of a key is 1.0 and movement is computed from 
;;; center of key to center of key thus the finger movement crosses half of
;;; the target key when striking it).  This affects when the output-key
;;; action (and the call to rpm-window-key-event handler) occurs.  The default
;;; motor module assumes that the action occurs after the entire movement
;;; occurs and ignores the key closure time which is what will still happen if 
;;; the value is set to 0.  
;;; If it is set to a non-zero value then the peck and peck-recoil actions
;;; will generate the output-key action at the time: ct + (ft*(d-ps)/d) where
;;; ct is the key-closure-time, ft is the movement time computed from Fitts's 
;;; Law as usual, d is the distance of the movement, and ps is the value of
;;; the peck-strike-distance parameter.  Basically, it's a very rough 
;;; calculation that assumes once the finger is part way across the target 
;;; key it starts to depress and then registers after the closure-time passes.
;;; The assumption is that the travel time is lienar (which isn't right) and 
;;; thus the peck-strike-distance is subtracted from the distance across the 
;;; key to determine the proportion of the time which passes before contact
;;; is made.
;;;
;;; New Buffers for tracking hands: 
;;; 
;;; manual-left and manual-right
;;; 
;;; Each buffer allows for the same queries as the manual buffer except for
;;; the preparation query which it does not track.  The processor and execution
;;; queries of these buffers are busy if the corresponding hand is performing an
;;; action (or if either hand is when there aren't separate stages).  The state
;;; is busy if there is an action being prepared or if the processor or execution
;;; for this hand is busy.
;;; Each also adds additional queries for the fingers on that hand.  A finger
;;; query is busy if the finger is holding a key down or performing an action
;;; which affects the state of a key being held down, and free otherwise.
;;;
;;; Here is the output of the buffer-status call for the manual-left buffer for
;;; reference:
;;;
;;; MANUAL-LEFT:
;;;   buffer empty          : T
;;;   buffer full           : NIL
;;;   buffer requested      : NIL
;;;   buffer unrequested    : NIL
;;;   state free            : T
;;;   state busy            : NIL
;;;   state error           : NIL
;;;   processor free        : T
;;;   processor busy        : NIL
;;;   execution free        : T
;;;   execution busy        : NIL
;;;   last-command          : NONE
;;;   index  free           : T
;;;   middle free           : T
;;;   ring   free           : T
;;;   pinkie free           : T
;;;   thumb  free           : T
;;;
;;; Neither buffer does anything with requests which it receives nor are there any 
;;; chunks placed into either one.
;;;
;;;
;;;
;;; New request actions for the manual buffer.
;;;
;;; Here are low-level actions for directly manipulating the fingers:
;;;
;;; isa hold-punch
;;;  hand [left | right]
;;;  finger [index | middle | ring | pinkie | thumb]
;;; 
;;; Press and hold down the key under the indicated finger.  If the finger
;;; is already holding a key release it and depress it again.
;;;
;;;
;;; isa hold-peck
;;;  hand [left | right]
;;;  finger [index | middle | ring | pinkie | thumb]
;;;  r distance
;;;  theta direction
;;;
;;; Move the finger based on the distance and direction and hold down the
;;; key located there.  If the finger is currently holding a key release that
;;; key before performing this movement.
;;;
;;;
;;; isa release
;;;  hand [left | right]
;;;  finger [index | middle | ring | pinkie | thumb]
;;;
;;; Release the key held by the indicated finger and leave the finger
;;; over that key.  If the finger is not holding a key do nothing and
;;; output a warning.
;;;
;;;
;;; isa release-recoil
;;;  hand [left | right]
;;;  finger [index | middle | ring | pinkie | thumb]
;;;  r distance
;;;  theta direction
;;;
;;; Release the key held by the indicated finger and then move it the
;;; indicated distance and direction without striking the key at that
;;; location.  If the finger is not holding a key do nothing and output
;;; a warning.
;;;
;;;
;;; isa point-finger
;;;  hand [left | right]
;;;  finger [index | middle | ring | pinkie | thumb]
;;;  r distance
;;;  theta direction
;;;
;;; Move the finger based on the distance and direction without striking
;;; the key at that location.  If the finger is currently holding a key release that
;;; key before performing this movement.
;;;
;;; 
;;; Here are the new high-level actions that generate the corresponding low-level
;;; movement necessary.
;;;
;;;
;;; isa type-key
;;;  key key
;;;
;;; Move the appropriate finger to the key and press and release it.  The
;;; finger is moved from where it currently is (unlike press-key which assumes
;;; it's on the home position) and returns to where it was after striking the key
;;; (which may not be the home row position).
;;;
;;;
;;; isa hit-key
;;;  key key
;;;
;;; Move the appropriate finger to the key and press and release it.  The
;;; finger is moved from where it currently is (unlike press-key which assumes
;;; it's on the home position) and stays over the key which is hit.
;;;
;;;
;;; isa hold-key
;;;  key key
;;;
;;; Move the appropriate finger to the key and press it keeping it held down.  The
;;; finger is moved from where it currently is (unlike press-key which assumes
;;; it's on the home position).
;;;
;;;
;;; isa release-key
;;;  key key
;;;
;;; If the appropriate finger for that key is currently holding it down release it
;;; and leave the finger at that position.  If the finger is not holding down that
;;; key print a warning and do nothing.
;;;
;;;
;;; isa release-key-to-home
;;;  key key
;;;
;;; If the appropriate finger for that key is currently holding it down release it
;;; and move the finger back to its home position without striking the corresponding
;;; home key.  If the finger is not holding down that key print a warning and do nothing.
;;;
;;;
;;; isa move-to-key
;;;  key key
;;;
;;; Move the appropriate finger to that key without pressing it.  The finger is moved
;;; from where it currently is (unlike press-key which assumes it's on the home position).
;;;
;;;
;;; isa move-finger-to-home
;;;  hand [left | right]
;;;  finger [index | middle | ring | pinkie | thumb]
;;;
;;; Move the specified finger to its home position without pressing it.  If the finger
;;; is currently holding a key down release it before moving.  If the finger is already
;;; at the home position do nothing and print a warning.
;;;
;;; 
;;; isa all-fingers-to-home
;;;  {hand [left | right]}
;;;
;;; Move all fingers on the specified hand, or both hands if no hand specified,
;;; to their home positions.  Like move-finger-to-home any finger which is holding
;;; a key will be released first.  If all the fingers are on the home positions
;;; do nothing and print a warning.  This action has a fixed cost of 200ms.
;;;
;;;
;;; isa hold-mouse
;;;
;;; Execute a hold-punch with the right index finger to depress the mouse button.
;;; If the hand is not located on the mouse print a warning and do nothing.
;;;
;;;
;;; isa release-mouse
;;;
;;; Release the right index finger to release the mouse button.  If the 
;;; finger is not holding the button down or the hand is not located on the
;;; mouse do nothing and print a warning.
;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Design Choices:
;;; 
;;; Try to work with the existing motor module code as much as possible by
;;; subclassing and using after methods where possible.
;;;
;;; However, for the styles I've chosen to redefine the classes to add extra 
;;; slots since they're referred to by class name in the request function and
;;; the press-key mappings and don't want to make replacements for those instead.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; The code
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

#+:packaged-actr (in-package :act-r)
#+(and :clean-actr (not :packaged-actr) :ALLEGRO-IDE) (in-package :cg-user)
#-(or (not :clean-actr) :packaged-actr :ALLEGRO-IDE) (in-package :cl-user)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Define a new method for the device which "releases keys" and specify new
;;; event handlers for that action on both the keys and clicks:
;;;
;;; rpm-window-click-release-event-handler
;;; rpm-window-key-release-event-handler
;;;
;;; Those event handlers can be defined by the modeler on the device for use with
;;; model actions only.  There are no mechanisms for the real windows to pass 
;;; that information along directly because this doesn't go through a 
;;; "device-handle-release" method which would be the appropriate mechanism.
;;;


(defmethod release-key ((devin device-interface) (keyloc vector))
  (let* ((invalid (or (< (svref keyloc 1) 0)
                      (> (svref keyloc 1) 6)
                      (< (svref keyloc 0) 0)
                      (and (> (svref keyloc 0) 22)
                           (not (= (svref keyloc 0) 28)))))
         (the-key (if invalid nil (loc->key (keyboard devin) keyloc))))
    (indicate-model-generated 
     (if (eq the-key 'mouse)
         (rpm-window-click-release-event-handler (device devin) (get-mouse-coordinates (device devin)))
       (progn 
         (when (null the-key)
           (print-warning "Invalid key location released ~s" keyloc))
         (rpm-window-key-release-event-handler (device devin) the-key))))))

(defmethod rpm-window-click-release-event-handler ((device t) pos)
  (declare (ignorable device pos)))

(defmethod rpm-window-key-release-event-handler ((device t) key)
  (declare (ignorable device key)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Code to implement a replacement for the standard motor module
;;; which tracks the processor and execution stages separately for
;;; left and right hand actions.  There is still only one preparation
;;; stage for the whole module.
;;;
;;; Do this by subclassing the motor-module class, redefine the creation
;;; function for the motor module to instantiate this class instead, 
;;; and then define the necessary methods on that new class.
;;;
;;; The only functional difference is that now the proc-s, exec-s, and 
;;; exec-queue slots hold a list of two items (left and right) respectively
;;; instead of just one.
;;;
;;; Also adding slots for holding the new parameters that are available
;;; for configuring the operation and adjusting the timing of the
;;; keypress actions.


(defclass dual-execution-motor-module (motor-module) 
  ((key-closure-time :accessor key-closure-time)
   (key-release-time :accessor key-release-time)
   (two-exec :accessor two-exec)
   (two-proc :accessor two-proc)
   (peck-strike :accessor peck-strike :initform 0)))

;;; Redefine the creation function for the motor module to create the new class.

(defun create-motor-module (name)
  (declare (ignore name))
  (make-instance 'dual-execution-motor-module))

;;; Set necessary slots to lists after the regular methods happen

(defmethod clear :after ((module dual-execution-motor-module))
  (when (not (check-jam module))
    (setf (exec-queue module) (list nil nil))))

(defmethod reset-pm-module :after ((module dual-execution-motor-module))
  (setf (proc-s module) (list 'free 'free))
  (setf (exec-s module) (list 'free 'free))
  (setf (exec-queue module) (list nil nil)))

;;; When changing state consider which hand the action is using

(defmethod change-state ((module dual-execution-motor-module) &key proc exec prep last)
  (when proc 
    (if (listp proc)
        (cond ((or (null (two-proc module)) (eq (car proc) 'both))
               (setf (first (proc-s module)) (second proc))
               (setf (second (proc-s module)) (second proc)))
              ((eq (car proc) 'left)
               (setf (first (proc-s module)) (second proc)))
              ((eq (car proc) 'right)
               (setf (second (proc-s module)) (second proc)))
              (t 
               (print-warning "Invalid proc setting for the dual execution motor module: ~S" proc)))
      (print-warning "Invalid proc setting for the dual execution motor module: ~S" proc)))
  
  (when exec 
    (if (listp exec)
        (cond ((or (null (two-exec module)) (eq (car exec) 'both))
               (setf (first (exec-s module)) (second exec))
               (setf (second (exec-s module)) (second exec)))
              ((eq (car exec) 'left)
               (setf (first (exec-s module)) (second exec)))
              ((eq (car exec) 'right)
               (setf (second (exec-s module)) (second exec)))
              (t 
               (print-warning "Invalid exec setting for the dual execution motor module: ~S" exec)))
      (print-warning "Invalid exec setting for the dual execution motor module: ~S" exec)))
  
  (when prep (setf (prep-s module) prep))
  (when last (setf (last-cmd module) last))
  
  (if (or (eq (first (proc-s module)) 'busy) 
          (eq (second (proc-s module)) 'busy) 
          (eq (first (exec-s module)) 'busy)
          (eq (second (exec-s module)) 'busy)
          (eq (prep-s module) 'busy))
    (setf (mode-s module) 'busy)
    (setf (mode-s module) 'free))
  
  (setf (state-change module) t))

;;; For testing state the processor and execution are busy if either
;;; hand's stage is busy and free only if both hands' stages are free.

;;; Can't check based on hand with the manual buffer, but there are
;;; two additional buffers (manual-left and manual-right) which allow 
;;; for testing that if desired, and those buffers also work for 
;;; tracking BOLD data by hand.


(defmethod print-module-status ((mod dual-execution-motor-module))
  (command-output "  preparation free      : ~S" (eq (prep-s mod) 'free))
  (command-output "  preparation busy      : ~S" (eq (prep-s mod) 'busy))
  (command-output "  processor free        : ~S" (and (eq (first (proc-s mod)) 'free)
                                                      (eq (second (proc-s mod)) 'free)))
  (command-output "  processor busy        : ~S" (or (eq (first (proc-s mod)) 'busy)
                                                     (eq (second (proc-s mod)) 'busy)))
  (command-output "  execution free        : ~S" (and (eq (first (exec-s mod)) 'free)
                                                      (eq (second (exec-s mod)) 'free)))
  (command-output "  execution busy        : ~S" (or (eq (first (exec-s mod)) 'busy)
                                                     (eq (second (exec-s mod)) 'busy)))
  (command-output "  last-command          : ~S" (last-cmd mod)))

(defmethod generic-state-query ((module dual-execution-motor-module) buffer slot value)
  (case slot
    ((state modality)
     (case value
       (busy
        (eq (mode-s module) 'busy))
       (free
        (eq (mode-s module) 'free))
       (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value))))
    (execution
     (case value
       (busy
        (or (eq (first (exec-s module)) 'busy)
            (eq (second (exec-s module)) 'busy)))
       (free
        (and (eq (first (exec-s module)) 'free)
             (eq (second (exec-s module)) 'free)))
       (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value))))
    (preparation
     (case value
       (busy
        (eq (prep-s module) 'busy))
       (free
        (eq (prep-s module) 'free))
       (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value))))
    (processor
     (case value
       (busy
        (or (eq (first (proc-s module)) 'busy)
            (eq (second (proc-s module)) 'busy)))
       (free
        (and (eq (first (proc-s module)) 'free)
             (eq (second (proc-s module)) 'free)))
       (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value))))
    (last-command 
     (eql (last-cmd module) value))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Change the style mechanisms:
;;;
;;; Give the movement style the slots for indicating whether or not
;;; the movement requires two "executions" (a press and release) and 
;;; for specifying the state of the finger(s) involved at the start 
;;; and end of the action.
;;; 
;;; Add a method for computing the second execution time.
;;;
;;; Send the hand information to change-state.
;;;
;;; Make dummy requests to the manual-left and manual-right buffers
;;; as appropriate so that an action is recorded for the buffer trace.
;;;
;;; Indicate style and hand in the motor action events' output.
;;;
;;; The new slots are used as follows:
;;;
;;; two-exec-p defaults to nil, but if set to t indicates that the
;;; compute-second-exec-time method should be called.
;;;
;;; exec2-time stores the result of calling compute-second-exec-time.
;;;
;;; release-if-down indicates whether or not the finger used for
;;; the action should produce a release event if it is down when the
;;; execution of this action starts.  The possible values are:
;;; nil which means do not generage a release event.
;;; :free (the default) which means to generate the release event and
;;; not charge a time cost for it.
;;; :penalty generates the release event and adds a burst-time cost
;;; to each of the exec, exec2, and finish times of the action.
;;;
;;; set-finger-down indicates whether the action should mark the
;;; finger used as being busy when the execution starts.  The default
;;; is nil.
;;;
;;; set-finger-up indicates whether the action should mark the finger
;;; used as being free when the execution finishes.  The default is t.


(defclass movement-style ()
  ((fprep-time :accessor fprep-time :initform nil :initarg :fprep-time)
   (exec-time :accessor exec-time :initform nil :initarg :exec-time)
   (finish-time :accessor finish-time :initform nil :initarg :finish-time)
   (exec-immediate-p :accessor exec-immediate-p :initform t :initarg :exec-immediate-p)
   (num-features :accessor num-features :initform nil :initarg :num-features)
   (style-name :accessor style-name :initarg :style-name :initform nil)
   (feature-slots :accessor feature-slots :initarg :feature-slots :initform nil)
   
   (two-exec-p :accessor two-exec-p :initform nil :initarg :two-exec-p)
   (exec2-time :accessor exec2-time :initform 0 :initarg exec2-time)
   
   (release-if-down :accessor release-if-down :initarg :release-if-down :initform :free)
   (set-finger-down :accessor set-finger-down :initarg :set-finger-down :initform nil)
   (set-finger-up :accessor set-finger-up :initarg :set-finger-up :initform t)))


(defgeneric compute-second-exec-time (module movement)
  (:documentation "Return the release time of <movement>."))

(defmethod compute-second-exec-time ((module pm-module) (mvmt movement-style))
  (error "No method defined for COMPUTE-SECOND-EXEC-TIME."))


(defmethod prepare-movement ((module dual-execution-motor-module) (mvmt movement-style))
  (change-state module :prep 'BUSY :proc (list (hand mvmt) 'BUSY))
  (setf (fprep-time mvmt) 
        (rand-time (compute-prep-time module mvmt)))
  (setf (last-prep module) mvmt)
  (schedule-event-relative  (fprep-time mvmt) 'preparation-complete :destination (my-name module)
                           :module (my-name module) 
                           :details (format nil "PREPARATION-COMPLETE style ~s hand ~s" (type-of mvmt) (hand mvmt)))
  
  ;; send dummy requests so that the buffer trace records things appropriately by hand
  (when (or (eq (hand mvmt) 'left) (eq (hand mvmt) 'both))
    (schedule-event-relative 0 'module-request 
                             :module :motor
                             :params (list 'manual-left
                                           (define-chunk-spec-fct (list 'isa (if (chunk-type-p-fct (type-of mvmt))
                                                                                 (type-of mvmt)
                                                                               (chunk-type-fct (list (type-of mvmt)))))))
                             :details (concatenate 'string (symbol-name 'module-request) " " (symbol-name 'manual-left))
                             :output nil
                             :maintenance t))
  
  (when (or (eq (hand mvmt) 'right) (eq (hand mvmt) 'both))
    (schedule-event-relative 0 'module-request 
                             :module :motor
                             :params (list 'manual-right
                                           (define-chunk-spec-fct (list 'isa (if (chunk-type-p-fct (type-of mvmt))
                                                                                 (type-of mvmt)
                                                                               (chunk-type-fct (list (type-of mvmt)))))))
                             :details (concatenate 'string (symbol-name 'module-request) " " (symbol-name 'manual-right))
                             :output nil
                             :maintenance t)))



(defmethod initiation-complete ((module dual-execution-motor-module))
  (when (last-prep module)
    (case (hand (last-prep module))
      (left 
       (change-state module :proc '(left FREE)))
      (right 
       (change-state module :proc '(right FREE)))
      (both
       (change-state module :proc '(left FREE))
       (change-state module :proc '(right FREE)))
      (t
       (print-warning "Non-handed action taken with the two-handed motor module: ~s" (last-prep module))))))

(defmethod preparation-complete ((module dual-execution-motor-module))
  (change-state module :prep 'free)
  (when (last-prep module)
    (if (exec-immediate-p (last-prep module))
        (let ((hand (hand (last-prep module))))
          (cond ((or (null (two-exec module)) (eq hand 'left) (eq hand 'both)) 
                 (push-last (last-prep module) (first (exec-queue module))))
                ((eq hand 'right) 
                 (push-last (last-prep module) (second (exec-queue module))))
                (t
                 (print-warning "Non-handed action taken with the dual-execution motor module: ~s ~s" (last-prep module) (hand (last-prep module))))))
      
      (when (and (plusp (init-stamp module))
                 (>= (mp-time-ms) (+ (init-stamp module) (seconds->ms (init-time module)))))
        (change-state module :proc (list (hand (last-prep module)) 'FREE)))))
  (maybe-execute-movement module))

(defmethod execute ((module dual-execution-motor-module))
  (cond ((not (last-prep module))
         (model-warning "Motor Module has no movement to EXECUTE."))
        ((eq (prep-s module) 'BUSY)
         (model-warning "Motor Module cannot EXECUTE features being prepared."))
        (t
         (case (hand (last-prep module))
           ((left both)
            (setf (first (exec-queue module))
               (append (first (exec-queue module)) (mklist (last-prep module)))))
           (right 
            (setf (second (exec-queue module))
               (append (second (exec-queue module)) (mklist (last-prep module))))))
         (maybe-execute-movement module))))
       
(defmethod maybe-execute-movement ((module dual-execution-motor-module))
  ;; Check both exec queues everytime since we don't know which action
  ;; triggered this check.
  ;; The "left" queue also handles both hand actions and is the only
  ;; one used when there's not multiple exec stages.
  (when (and (first (exec-queue module)) (or (and (eq (hand (first (first (exec-queue module)))) 'left)
                                                  (eq (first (exec-s module)) 'FREE))
                                             (and (eq (hand (first (first (exec-queue module)))) 'both)
                                                  (eq (first (exec-s module)) 'FREE)
                                                  (eq (second (exec-s module)) 'FREE))
                                             (and (null (two-exec module))
                                                  (eq (first (exec-s module)) 'FREE))))
    (perform-movement module (pop (first (exec-queue module)))))
  
  (when (and (second (exec-queue module)) (eq (second (exec-s module)) 'FREE))
    (perform-movement module (pop (second (exec-queue module))))))


(defmethod perform-movement ((module dual-execution-motor-module) (mvmt movement-style))

  (schedule-event-relative  (init-time module) 'INITIATION-COMPLETE :destination (my-name module)
                           :module (my-name module) 
                           :details (format nil "INITIATION-COMPLETE style ~s hand ~s" (type-of mvmt) (hand mvmt)))
  
  (change-state module :proc (list (hand mvmt) 'BUSY) :exec (list (hand mvmt) 'BUSY))
  
  (setf (init-stamp module) (mp-time-ms))
  
  (setf (exec-time mvmt) (compute-exec-time module mvmt))
  (setf (finish-time mvmt) (compute-finish-time module mvmt))
  (when (two-exec-p mvmt)
    (setf (exec2-time mvmt) (compute-second-exec-time module mvmt)))
  
  
  ;;; Automatically release the fingers involved in an action
  ;;; and charge a burst-time cost if they're down at the start
  ;;; of the action and the style says they need to be released.
  
  (when (release-if-down mvmt)
    
    (let ((hands (if (eq (hand mvmt) 'both) (list 'left 'right) (list (hand mvmt))))
          (fingers (let ((f (when (find 'finger (feature-slots mvmt)) (finger mvmt)))) (if (and f (not (eq f :dummy))) (list f) '(index middle ring pinkie thumb))))
          (adjust nil))
      
      (dolist (hand hands)
        (dolist (finger fingers)
          (when (finger-down hand finger)
            (setf adjust t)
            (schedule-event-relative (+ (init-time module) (key-release-time module))
                                     'release-key :destination :device :module :motor :output 'medium
                                     :params (list (finger-loc-m module hand finger))))))
      (when (and adjust (eq (release-if-down mvmt) :penalty))
        (incf (exec-time mvmt) (burst-time module))
        (incf (finish-time mvmt) (burst-time module))
        (when (two-exec-p mvmt)
          (incf (exec2-time mvmt) (burst-time module))))))
  
  (queue-output-events module mvmt)
  (queue-finish-event module mvmt)
  
  ;;; If the style says to mark that its fingers are down do so now

  (when (set-finger-down mvmt)
    (let ((hands (if (eq (hand mvmt) 'both) (list 'left 'right) (list (hand mvmt))))
          (fingers (let ((f (when (find 'finger (feature-slots mvmt)) (finger mvmt)))) (if (and f (not (eq f :dummy))) (list f) '(index middle ring pinkie thumb)))))
      (dolist (hand hands)
        (dolist (finger fingers)
          (setf (gethash (list hand finger) (hand-tracker-finger-tracker (get-module motor-extension))) t))))))



(defmethod queue-finish-event ((module dual-execution-motor-module) (mvmt movement-style))
  (schedule-event-relative (finish-time mvmt) 'finish-movement-dual :destination (my-name module)
                           :params (list mvmt) :module (my-name module) 
                           :details (format nil "FINISH-MOVEMENT style ~s hand ~s" (type-of mvmt) (hand mvmt))))


(defmethod finish-movement-dual ((module dual-execution-motor-module) (mvmt movement-style))
  (change-state module :exec (list (hand mvmt) 'free))
  
  ;;; If the style says to remove the finger down marker do so now
  (when (set-finger-up mvmt)
    (let ((hands (if (eq (hand mvmt) 'both) (list 'left 'right) (list (hand mvmt))))
          (fingers (let ((f (when (find 'finger (feature-slots mvmt)) (finger mvmt)))) (if (and f (not (eq f :dummy))) (list f) '(index middle ring pinkie thumb)))))
      (dolist (hand hands)
        (dolist (finger fingers)
          (remhash (list hand finger) (hand-tracker-finger-tracker (get-module motor-extension)))))))
  
  (maybe-execute-movement module))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Code for recording and testing hand execution with separate buffers.
;;;
;;; With the new buffers the fingers can also be tested to determine if 
;;; they're currently "holding" a key -- if the finger is free it is 
;;; not holding a key and if it is busy then it is either holding a key 
;;; or performing some other action related to holding or releasing a key.


(defstruct hand-tracker 
  manual left-cmd right-cmd (finger-tracker (make-hash-table :size 10 :test 'equalp)))

(defun track-hand-requests (module buffer spec)
  (if (eq buffer 'manual-left)
      (setf (hand-tracker-left-cmd module) (chunk-spec-chunk-type spec))
    (setf (hand-tracker-right-cmd module) (chunk-spec-chunk-type spec))))

(defun reset-hand-tracker (module)
  (setf (hand-tracker-manual module) (get-module :motor))
  (setf (hand-tracker-left-cmd module) 'none)
  (setf (hand-tracker-right-cmd module) 'none)
  (clrhash (hand-tracker-finger-tracker module)))
  
(defun finger-down (hand finger)
  (gethash (list hand finger) (hand-tracker-finger-tracker (get-module motor-extension))))

(defmethod print-hand-buffer-status ((mod hand-tracker) hand)
  (let ((manual (hand-tracker-manual mod)))
    (command-output "  processor free        : ~S"
                    (if (eq hand 'manual-left)
                        (eq (first (proc-s manual)) 'free)
                      (eq (second (proc-s manual)) 'free)))
    (command-output "  processor busy        : ~S"
                    (if (eq hand 'manual-left)
                        (eq (first (proc-s manual)) 'busy)
                      (eq (second (proc-s manual)) 'busy)))
    (command-output "  execution free        : ~S"
                    (if (eq hand 'manual-left)
                        (eq (first (exec-s manual)) 'free)
                      (eq (second (exec-s manual)) 'free)))
    (command-output "  execution busy        : ~S"
                    (if (eq hand 'manual-left)
                        (eq (first (exec-s manual)) 'busy)
                      (eq (second (exec-s manual)) 'busy)))
    (command-output "  last-command          : ~S"
                    (if (eq hand 'manual-left)
                        (hand-tracker-left-cmd mod)
                      (hand-tracker-right-cmd mod)))
    (dolist (x '(index middle ring pinkie thumb))
      (command-output "  ~(~6a~) free           : ~s" x
                      (null (finger-down (if (eq hand 'manual-left) 'left 'right) x))))))

(defmethod generic-state-query ((module hand-tracker) buffer slot value)
  (let ((manual (hand-tracker-manual module)))
    (case slot
      (preparation ;; same for both hands
       (case value
         (busy
          (eq (prep-s module) 'busy))
         (free
          (eq (prep-s module) 'free))
         (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value))))
      
      (state
       (case value
         (busy (case buffer
                 (manual-left
                  (or (eq (first (exec-s manual)) 'busy) (eq (first (proc-s manual)) 'busy)))
                 (manual-right
                  (or (eq (second (exec-s manual)) 'busy) (eq (second (proc-s manual)) 'busy)))))
         (free (case buffer
                 (manual-left
                  (and (eq (first (exec-s manual)) 'free) (eq (first (proc-s manual)) 'free)))
                 (manual-right
                  (and (eq (second (exec-s manual)) 'free) (eq (second (proc-s manual)) 'free)))))
         (error nil)
         (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value))))
      (execution
       (case value
         (busy (case buffer
                 (manual-left
                  (eq (first (exec-s manual)) 'busy))
                 (manual-right
                  (eq (second (exec-s manual)) 'busy))))
         (free (case buffer
                 (manual-left
                  (eq (first (exec-s manual)) 'free))
                 (manual-right
                  (eq (second (exec-s manual)) 'free))))
         (error nil)
         (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value))))
      
      (processor
       (case value
         (busy (case buffer
                 (manual-left
                  (eq (first (proc-s manual)) 'busy))
                 (manual-right
                  (eq (second (proc-s manual)) 'busy))))
         (free (case buffer
                 (manual-left
                  (eq (first (proc-s manual)) 'free))
                 (manual-right
                  (eq (second (proc-s manual)) 'free))))
         (error nil)
         (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value))))
      (last-command 
       (case buffer
         (manual-left
          (eql (hand-tracker-left-cmd module) value))
         (manual-right
          (eql (hand-tracker-right-cmd module) value))))
      ((index middle ring pinkie thumb)
       (case value
         (busy
          (gethash (list (if (eq buffer 'manual-left) 'left 'right) slot) (hand-tracker-finger-tracker module)))
         (free
          (null (gethash (list (if (eq buffer 'manual-left) 'left 'right) slot) (hand-tracker-finger-tracker module))))
         (t (print-warning "Invalid query made of the ~S buffer with slot ~S and value ~S" buffer slot value)))))))


(defun track-hand-params (instance param)
  ;; The parameter values are actually stored in the motor module
  ;; and not the extension module.
  (let ((manual (aif (hand-tracker-manual instance) it (get-module :motor))))
    (cond ((consp param)
          (case (car param)
            (:key-closure-time
             (setf (key-closure-time (current-device-interface)) (cdr param))
             (setf (key-closure-time manual) (cdr param)))
            (:key-release-time
             (setf (key-release-time manual) (cdr param)))
            (:dual-processor-stages
             (setf (two-proc manual) (cdr param)))
            (:dual-execution-stages
             (setf (two-exec manual) (cdr param)))
            (:peck-strike-distance
             (setf (peck-strike manual) (cdr param)))))
         (t
          (case param
            (:key-closure-time
             (key-closure-time manual))
            (:key-release-time
             (key-release-time manual))
            (:dual-processor-stages
             (two-proc manual))
            (:dual-execution-stages
             (two-exec manual))
            (:peck-strike-distance
             (peck-strike manual)))))))

(define-module-fct 'motor-extension 
    (list (list 'manual-left nil nil '(execution processor last-command index middle ring pinkie thumb)
                (lambda () 
                  (print-hand-buffer-status (get-module motor-extension) 'manual-left)))
          
          (list 'manual-right nil nil '(execution processor last-command index middle ring pinkie thumb)
                (lambda () 
                  (print-hand-buffer-status (get-module motor-extension) 'manual-right)))) 
  (list
   (define-parameter :key-closure-time
       :valid-test 'nonneg
       :warning "a number"
       :default-value 0.01
       :documentation "Time between when a key is contacted and it registers as being hit.")
   (define-parameter :key-release-time
       :valid-test 'nonneg
       :warning "a number"
       :default-value 0.04
     :documentation "Time between when a key that is held down is released and when it registers as being released.")
   (define-parameter :dual-processor-stages
       :valid-test 'tornil
       :warning "T or nil"
       :default-value nil
     :documentation "Whether the motor module has separate processor stages for the hands.")
   (define-parameter :dual-execution-stages
       :valid-test 'tornil
       :warning "T or nil"
       :default-value nil
     :documentation "Whether the motor module has separate execution stages for the hands.")
   (define-parameter :peck-strike-distance
       :valid-test (lambda (x) (<= 0 x .5))
       :warning "a number between 0 and .5"
       :default-value 0.0
       :documentation "Distance from center of the key at which time a peck starts to depress the key."))
  
  :params 'track-hand-params
  :request 'track-hand-requests
  :query 'generic-state-query 
  :creation (lambda (x) (make-hand-tracker))
  :reset (list nil nil 'reset-hand-tracker)
  :version "1.0a1" :documentation "Extends motor module with dual processor and/or execution states and finger holding actions.")


;;; Compute when a peck or peck-recoil indicates that the key is hit
;;; based on the peck-strike-distance paramter.
;;; 
;;; For the peck and peck-recoil compute the timing a little differently
;;; now because I want to consider the time between striking and releasing
;;; the key.  So, send the output-key before the movement time is "done"
;;; to simulate the key-closure-time as punch does i.e. the key gets 
;;; detected before the motion is complete.  To do that I'm assuming that
;;; the key starts to move once the finger has been moving for tt*(r-d)/r
;;; where tt is the computed travel time, r is the length of the move, and
;;; d is the peck-strike-distance.
;;; Basically, it's a very rough calculation that just assumes once the finger
;;; is part way across the target key it starts to depress and then registers
;;; after the closure-time passes.  The assumption is that the travel time
;;; is lienar (which isn't right) and that the strike-distance is subtracted
;;; from the distance across the target (which would be at most .5 since the
;;; default width of the key targets is 1.0 and it's going to the middle).
;;;

(defun movement-strike-time (mtr-mod tt r)
  (if (zerop (peck-strike mtr-mod))
      tt
    (ms-round (+ (key-closure-time mtr-mod)
                 (* tt (/ (- r (peck-strike mtr-mod)) r))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Redefine the standard motor actions: punch, peck, and peck-recoil
;;; to deal with calling the release event and indicate that they
;;; should automatically release the key (paying a burst-time cost) and 
;;; indicate the finger as busy during the action.
;;;
;;; The release time is determined to be key-release-time after the 
;;; "striking" action is over i.e. the burst or movement time.  So it
;;; happens somewhere during the final burst-time.

;;; Create a base class for hand and finger movements since there are
;;; other related styles now instead of only punches.

(defstyle hf-movement () hand finger)

(defclass punch (hf-movement)
  nil
  (:default-initargs
    :style-name :PUNCH
    :two-exec-p t
    :release-if-down :penalty
    :set-finger-down t))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self punch))
  (+ (init-time mtr-mod) 
     (key-closure-time mtr-mod)))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self punch))
  (+ (init-time mtr-mod) 
     (burst-time mtr-mod)
     (burst-time mtr-mod)))

(defmethod compute-second-exec-time ((mtr-mod dual-execution-motor-module) (self punch))
  (+ (init-time mtr-mod) 
     (burst-time mtr-mod)
     (key-release-time mtr-mod)))

(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self punch))
  
  (schedule-event-relative (exec-time self)
                           'output-key :destination :device :module :motor :output 'medium
                           :params (list (finger-loc-m mtr-mod (hand self) (finger self))))

  (schedule-event-relative (exec2-time self)
                           'release-key :destination :device :module :motor :output 'medium
                           :params (list (finger-loc-m mtr-mod (hand self) (finger self)))))


;; add a move-time slot to pecks to save the calculation

(defclass peck (hfrt-movement)
  ((move-time :accessor move-time))
  (:default-initargs
    :style-name :PECK
    :two-exec-p t
    :release-if-down :penalty
    :set-finger-down t))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self peck))
  (setf (move-time self)
    (max (burst-time mtr-mod)
         (rand-time (fitts mtr-mod (peck-fitts-coeff mtr-mod) (r self)))))
  
  (+ (init-time mtr-mod)
     (movement-strike-time mtr-mod  (move-time self) (r self))))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self peck))
  (+ (init-time mtr-mod) 
     (move-time self) 
     (burst-time mtr-mod)))

(defmethod compute-second-exec-time ((mtr-mod dual-execution-motor-module) (self peck))
  (+ (init-time mtr-mod) 
     (move-time self)
     (key-release-time mtr-mod)))

(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self peck))
  
  (schedule-event-relative (exec-time self) 
                           'output-key :destination :device :module :motor :output 'medium
                           :params (list (move-a-finger mtr-mod (hand self) (finger self) (r self) (theta self))))
  
  (schedule-event-relative (exec2-time self)
                           'release-key :destination :device :module :motor :output 'medium
                           :params (list (finger-loc-m mtr-mod (hand self) (finger self)))))



;; for a peck-recoil record the base movement time and the specific (noisy) first movement times

(defclass peck-recoil (hfrt-movement)
  ((move-time :accessor move-time)
   (first-move-time :accessor first-move-time))
  (:default-initargs
    :style-name :PECK
    :two-exec-p t
    :release-if-down :penalty
    :set-finger-down t))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self peck-recoil))
  (setf (move-time self)
    (max (burst-time mtr-mod)
         (fitts mtr-mod (peck-fitts-coeff mtr-mod) (r self))))
  
  (setf (first-move-time self)
    (max (burst-time mtr-mod) 
         (rand-time (move-time self))))
  
  (+ (init-time mtr-mod)
     (movement-strike-time mtr-mod (first-move-time self) (r self))))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self peck-recoil))
  (+ (init-time mtr-mod) 
     (first-move-time self) 
     (burst-time mtr-mod) 
     (max (burst-time mtr-mod) (rand-time (move-time self)))))

(defmethod compute-second-exec-time ((mtr-mod dual-execution-motor-module) (self peck-recoil))
  (+ (init-time mtr-mod) 
     (first-move-time self)
     (key-release-time mtr-mod)))

(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self peck-recoil))
    
  (schedule-event-relative (exec-time self) 'output-key 
                           :destination :device :module :motor :output 'medium
                           :params (list (polar-move-xy (finger-loc-m mtr-mod (hand self) (finger self))
                                                        (vector (r self) (theta self)))))
  
  (schedule-event-relative (exec2-time self)
                           'release-key :destination :device :module :motor :output 'medium
                           :params (list (polar-move-xy (finger-loc-m mtr-mod (hand self) (finger self))
                                                        (vector (r self) (theta self))))))


;;; Make sure that mouse movement doesn't change the state of the
;;; mouse button

(defclass cursor-ply (ply)
  ((target-coords :accessor target-coords :initarg :target-coords)
   (control-order :accessor control-order :initarg :control-order :initform 0))
  (:default-initargs
    :hand 'RIGHT
    :fitts-coeff 
    (when (current-device-interface)
      (mouse-fitts-coeff (current-device-interface )))
    :release-if-down nil
    :set-finger-up nil))

;;; For the new actions use this function for processing all of
;;; the requests instead of writing a new one for each.  This
;;; should also be how the regular module works too instead of
;;; all the repeated code in the request function...

(defmethod handle-extended-style ((mtr-mod dual-execution-motor-module) chunk-spec)
  (let* ((style (chunk-spec-chunk-type chunk-spec))
         (dummy (make-instance style))
         (params-list (mapcan (lambda (x)
                                (aif (and (slot-in-chunk-spec-p chunk-spec x) 
                                          (verify-single-explicit-value 
                                           (chunk-spec-slot-spec chunk-spec x) 
                                           :motor style x))
                                     (list (sym->key x) it)
                                     (return-from handle-extended-style nil)))
                        (feature-slots dummy))))
    (schedule-event-relative 0 style :destination :motor :params params-list :module :motor :output 'low)))


;;; New styles for actions which hold a finger down or release a held finger

(defclass hold-punch (punch)
  nil
  (:default-initargs
    :style-name :PUNCH
    :release-if-down :penalty
    :set-finger-down t
    :set-finger-up nil))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self hold-punch))
  (+ (init-time mtr-mod) 
     (key-closure-time mtr-mod)))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self hold-punch))
  (+ (init-time mtr-mod) 
     (burst-time mtr-mod)))

(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self hold-punch))
  (schedule-event-relative (exec-time self)
                           'output-key :destination :device :module :motor :output 'medium
                           :params (list (finger-loc-m mtr-mod (hand self) (finger self)))))

(defmethod hold-punch ((mtr-mod dual-execution-motor-module) &key hand finger)
  (unless (or (check-jam mtr-mod) (check-specs hand finger))
    (prepare-movement mtr-mod (make-instance 'hold-punch :hand hand :finger finger))))

(extend-manual-requests (hold-punch hand finger) handle-extended-style)


(defclass hold-peck (peck)
  nil
  (:default-initargs
    :style-name :PECK
    :release-if-down :penalty
    :set-finger-down t
    :set-finger-up nil))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self hold-peck))
  (setf (move-time self)
    (max (burst-time mtr-mod)
         (rand-time (fitts mtr-mod (peck-fitts-coeff mtr-mod) (r self)))))
  
  (+ (init-time mtr-mod)
     (movement-strike-time mtr-mod (move-time self) (r self))))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self hold-peck))
  (+ (init-time mtr-mod) 
     (move-time self)))
     
(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self hold-peck))
  (schedule-event-relative (exec-time self) 
                           'output-key :destination :device :module :motor :output 'medium
                           :params (list (move-a-finger mtr-mod (hand self) (finger self) (r self) (theta self)))))

(defmethod hold-peck ((mtr-mod motor-module) &key hand finger r theta)
  (unless (or (check-jam mtr-mod) (check-specs hand finger r theta))
    (when (symbolp theta)
      (setf theta (symbol-value theta)))
    (prepare-movement mtr-mod (make-instance 'hold-peck :hand hand :finger finger :r r :theta theta))))

(extend-manual-requests (hold-peck hand finger r theta) handle-extended-style)


(defclass release (punch)
  nil
  (:default-initargs
      :style-name :PUNCH
    :release-if-down nil))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self release))
  (+ (init-time mtr-mod) 
     (key-release-time mtr-mod)))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self release))
  (+ (init-time mtr-mod) 
     (burst-time mtr-mod)))

(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self release))
  (schedule-event-relative (exec-time self)
                           'release-key :destination :device :module :motor :output 'medium
                           :params (list (finger-loc-m mtr-mod (hand self) (finger self)))))

(defmethod release ((mtr-mod dual-execution-motor-module) &key hand finger)
  (unless (or (check-jam mtr-mod) (check-specs hand finger))
    (if (finger-down hand finger)
        (prepare-movement mtr-mod (make-instance 'release :hand hand :finger finger))
      (print-warning "RELEASE action ignored because the ~s ~s finger is not held down." hand finger))))

(extend-manual-requests (release hand finger) handle-extended-style)


(defclass release-recoil (peck)
  nil
  (:default-initargs
    :style-name :PECK
    :release-if-down nil))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self release-recoil))
  (+ (init-time mtr-mod) 
     (key-release-time mtr-mod)))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self release-recoil))
  (setf (move-time self)
    (max (burst-time mtr-mod)
         (rand-time (fitts mtr-mod (peck-fitts-coeff mtr-mod) (r self)))))
  
  (+ (init-time mtr-mod) 
     (move-time self) 
     (burst-time mtr-mod)))
               
(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self release-recoil))
  (schedule-event-relative (exec-time self)
                           'release-key :destination :device :module :motor :output 'medium
                           :params (list (finger-loc-m mtr-mod (hand self) (finger self))))
  ;; cheat a little like other styles and put the finger in the final position ahead of time
  (move-a-finger mtr-mod (hand self) (finger self) (r self) (theta self)))
  
(defmethod release-recoil ((mtr-mod dual-execution-motor-module) &key hand finger r theta)
  (unless (or (check-jam mtr-mod) (check-specs hand finger r theta))
    (when (symbolp theta)
      (setf theta (symbol-value theta)))
    (if (finger-down hand finger)
        (prepare-movement mtr-mod (make-instance 'release-recoil :hand hand :finger finger :r r :theta theta))
      (print-warning "RELEASE-RECOIL action ignored because the ~s ~s finger is not held down." hand finger))))

(extend-manual-requests (release-recoil hand finger r theta) handle-extended-style)  


;; Style for moving a finger without striking the key

(defclass point-finger (peck)
  nil
  (:default-initargs
      :style-name :PECK
      :release-if-down :penalty))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self point-finger))
  
  ;;; there isn't really an exec time needed so just return init-time
  (init-time mtr-mod))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self point-finger))
  (setf (move-time self)
    (max (burst-time mtr-mod)
         (rand-time (fitts mtr-mod (peck-fitts-coeff mtr-mod) (r self)))))
  
  (+ (init-time mtr-mod) 
     (move-time self)))
                                                                             
(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self point-finger))
  ;; just move the finger, there's nothing to schedule
  (move-a-finger mtr-mod (hand self) (finger self) (r self) (theta self)))

(defmethod point-finger ((mtr-mod motor-module) &key hand finger r theta)
  (unless (or (check-jam mtr-mod) (check-specs hand finger r theta))
    (when (symbolp theta)
      (setf theta (symbol-value theta)))
    (prepare-movement mtr-mod (make-instance 'point-finger :hand hand :finger finger :r r :theta theta))))

(extend-manual-requests (point-finger hand finger r theta) handle-extended-style)


;;; These are the new "macro" actions which take a key name and then generate 
;;; the appropriate low-level style as needed based on current finger position and
;;; spcified target. 

;;; Start with some functions that return useful keyboard information.

;;; Determine which hand and finger is used to strike the key based on the
;;; touch typing commands predefined for the keyboad.

(defun key->hand&finger (key)
  (let ((cmd (key-to-command (keyboard (current-device-interface)) key)))
    (if cmd
        (values (nth (1+ (position :hand cmd)) cmd) (nth (1+ (position :finger cmd)) cmd))
      (print-warning "No hand and finger mapping available for key ~s" key))))

;;; Indicate each finger's home position (which should really be in the keyboard
;;; definition instead of implicit in the module startup & hand-to-home actions).

(defun home-position (hand finger)
  (case hand
    (left (case finger
            (index #(4 4))
            (middle #(3 4))
            (ring #(2 4))
            (pinkie #(1 4))
            (thumb #(5 6))))
    (right (case finger
             (index #(7 4))
             (middle #(8 4))
             (ring #(9 4))
             (pinkie #(10 4))
             (thumb #(6 6))))))


;; type-key
;; Similar to the press-key command except that it doesn't assume the finger
;; starts from the home row.  It uses the appropriate finger's current position
;; to determine the movement necessary to strike the key and return to where it
;; started (which may or may not be the home row).

(defmethod type-key ((mtr-mod motor-module) spec)
  (let ((key (and (slot-in-chunk-spec-p spec 'key) 
                  (verify-single-explicit-value (chunk-spec-slot-spec spec 'key) :motor 'hit-key 'key))))
    (when key
      (when (stringp key)
        (setf key (read-from-string key)))
      (multiple-value-bind (hand finger) (key->hand&finger key)
        (when (and hand finger)
          (let ((key-loc (key-to-loc (keyboard (current-device-interface)) key))
                (finger-loc (finger-loc-m mtr-mod hand finger)))
            (if (vpt= key-loc finger-loc)
                (punch mtr-mod :hand hand :finger finger)
              (let ((vector (xy-to-polar finger-loc key-loc)))
                (peck-recoil mtr-mod :hand hand :finger finger :r (vr vector) :theta (vtheta vector))))))))))

(extend-manual-requests (type-key key) type-key)


;; hit-key
;; Similar to the press-key command except that it doesn't assume the finger
;; starts from the home row (it uses the current position) and it stays at the
;; key which was hit instead of returning to the home position.

(defmethod hit-key ((mtr-mod motor-module) spec)
  (let ((key (and (slot-in-chunk-spec-p spec 'key) 
                  (verify-single-explicit-value (chunk-spec-slot-spec spec 'key) :motor 'hit-key 'key))))
    (when key
      (when (stringp key)
        (setf key (read-from-string key)))
      (multiple-value-bind (hand finger) (key->hand&finger key)
        (when (and hand finger)
          (let ((key-loc (key-to-loc (keyboard (current-device-interface)) key))
                (finger-loc (finger-loc-m mtr-mod hand finger)))
            (if (vpt= key-loc finger-loc)
                (punch mtr-mod :hand hand :finger finger)
              (let ((vector (xy-to-polar finger-loc key-loc)))
                (peck mtr-mod :hand hand :finger finger :r (vr vector) :theta (vtheta vector))))))))))

(extend-manual-requests (hit-key key) hit-key)


;; hold-key
;; Like the hit-key action except that it continues to hold the key that is struck.

(defmethod hold-key ((mtr-mod motor-module) spec)
  (let ((key (and (slot-in-chunk-spec-p spec 'key) 
                  (verify-single-explicit-value (chunk-spec-slot-spec spec 'key) :motor 'hold-key 'key))))
    (when key
      (when (stringp key)
        (setf key (read-from-string key)))
      (multiple-value-bind (hand finger) (key->hand&finger key)
        (when (and hand finger)
          (let ((key-loc (key-to-loc (keyboard (current-device-interface)) key))
                (finger-loc (finger-loc-m mtr-mod hand finger)))
            (if (vpt= key-loc finger-loc)
                (hold-punch mtr-mod :hand hand :finger finger)
              (let ((vector (xy-to-polar finger-loc key-loc)))
                (hold-peck mtr-mod :hand hand :finger finger :r (vr vector) :theta (vtheta vector))))))))))

(extend-manual-requests (hold-key key) hold-key)


;; release-key
;; If the designated key is being held down release it and leave the finger over it.

(defmethod release-key ((mtr-mod motor-module) spec)
  (let ((key (and (slot-in-chunk-spec-p spec 'key) 
                  (verify-single-explicit-value (chunk-spec-slot-spec spec 'key) :motor 'release-key 'key))))
    (when key
      (when (stringp key)
        (setf key (read-from-string key)))
      (multiple-value-bind (hand finger) (key->hand&finger key)
        (when (and hand finger)
          (let ((key-loc (key-to-loc (keyboard (current-device-interface)) key))
                (finger-loc (finger-loc-m mtr-mod hand finger)))
            (if (and (vpt= key-loc finger-loc) (finger-down hand finger))
                (release mtr-mod :hand hand :finger finger)
              (print-warning "RELEASE-KEY cannot release the ~s key because it is not being held." key))))))))

(extend-manual-requests (release-key key) release-key)


;; release-key-to-home
;; If the designated key is being held down release it and return the finger to the
;; home position.

(defmethod release-key-to-home ((mtr-mod motor-module) spec)
  (let ((key (and (slot-in-chunk-spec-p spec 'key) 
                  (verify-single-explicit-value (chunk-spec-slot-spec spec 'key) :motor 'release-key-to-home 'key))))
    (when key
      (when (stringp key)
        (setf key (read-from-string key)))
      (multiple-value-bind (hand finger) (key->hand&finger key)
        (when (and hand finger)
          (let ((key-loc (key-to-loc (keyboard (current-device-interface)) key))
                (finger-loc (finger-loc-m mtr-mod hand finger)))
            (if (and (vpt= key-loc finger-loc) (finger-down hand finger))
                (let* ((home-pos (home-position hand finger))
                       (vector (xy-to-polar finger-loc home-pos)))
                  (release-recoil mtr-mod :hand hand :finger finger :r (vr vector) :theta (vtheta vector)))
              (print-warning "RELEASE-KEY-TO-HOME cannot release the ~s key because it is not being held." key))))))))

(extend-manual-requests (release-key-to-home key) release-key-to-home)


;; move-to-key
;; Position the appropriate finger over the designated key without striking it.

(defmethod move-to-key ((mtr-mod motor-module) spec)
  (let ((key (and (slot-in-chunk-spec-p spec 'key) 
                  (verify-single-explicit-value (chunk-spec-slot-spec spec 'key) :motor 'move-to-key 'key))))
    (when key
      (when (stringp key)
        (setf key (read-from-string key)))
      (multiple-value-bind (hand finger) (key->hand&finger key)
        (when (and hand finger)
          (let ((key-loc (key-to-loc (keyboard (current-device-interface)) key))
                (finger-loc (finger-loc-m mtr-mod hand finger)))
            (if (vpt= key-loc finger-loc)
                (print-warning "MOVE-TO-KEY action ignored because finger already over key ~s." key)
              (let ((vector (xy-to-polar finger-loc key-loc)))
                (point-finger mtr-mod :hand hand :finger finger :r (vr vector) :theta (vtheta vector))))))))))

(extend-manual-requests (move-to-key key) move-to-key)


;; move-finger-to-home 
;; Return a specified finger to the home position without striking that key.

(defmethod move-finger-to-home ((mtr-mod motor-module) spec)
  (let ((hand (and (slot-in-chunk-spec-p spec 'hand) 
                   (verify-single-explicit-value (chunk-spec-slot-spec spec 'hand) :motor 'move-finger-to-home 'hand)))
        (finger (and (slot-in-chunk-spec-p spec 'finger) 
                     (verify-single-explicit-value (chunk-spec-slot-spec spec 'finger) :motor 'move-finger-to-home 'finger))))
    (when (and hand finger)
      (let ((home-pos (home-position hand finger))
            (finger-loc (finger-loc-m mtr-mod hand finger)))
        (if (vpt= home-pos finger-loc)
            (print-warning "MOVE-FINGER-TO-HOME ignored because the ~s ~s is home." hand finger)
          (let ((vector (xy-to-polar finger-loc home-pos)))
            (point-finger mtr-mod :hand hand :finger finger :r (vr vector) :theta (vtheta vector))))))))
             
(extend-manual-requests (move-finger-to-home hand finger) move-finger-to-home)


;;; all-home
;;; A style to move all of the fingers on one or both hands to the home row.
;;; The cost for the action is fixed at 200ms.  However, the style itself
;;; isn't actually made available as a model action.  It's only available
;;; through the all-fingers-to-home action below.

(defclass all-home (hfrt-movement)
  nil
  (:default-initargs
    :style-name :all-home
    :hand 'both))

(defmethod compute-exec-time ((mtr-mod dual-execution-motor-module) (self all-home))
  ;; no exec time needed so just return init-time
  (init-time mtr-mod))

(defmethod compute-finish-time ((mtr-mod dual-execution-motor-module) (self all-home))
  (+ (init-time mtr-mod) 
     (max 
      (min-fitts-time mtr-mod)
      (rand-time .200))))

(defmethod queue-output-events ((mtr-mod dual-execution-motor-module) (self all-home))
  ;; just move them now...
  (dolist (hand (if (eq (hand self) 'both) (list 'left 'right) (list (hand self))))
    (dolist (finger '(index middle ring pinkie thumb))
      (let ((home-pos (home-position hand finger))
            (finger-loc (finger-loc-m mtr-mod hand finger)))
        (unless (vpt= home-pos finger-loc)
          (let ((vector (xy-to-polar finger-loc home-pos)))
            (move-a-finger mtr-mod hand finger (vr vector) (vtheta vector))))))))

(defmethod all-home ((mtr-mod motor-module) hand)
  (unless (check-jam mtr-mod) 
    (prepare-movement mtr-mod (make-instance 'all-home :hand hand))))


;; all-fingers-to-home
;; Return all the fingers on the specified hand (or both) to the home positions
;; without striking any keys.

(defmethod all-fingers-to-home ((mtr-mod motor-module) spec)
  (let* ((slot-specs (and (slot-in-chunk-spec-p spec 'hand) (chunk-spec-slot-spec spec 'hand)))
         (hands (cond ((zerop (length slot-specs)) '(left right))
                      ((and (= (length slot-specs) 1)
                            (eql '= (caar slot-specs))
                            (or (eq (third (car slot-specs)) 'left)
                                (eq (third (car slot-specs)) 'right)))
                       (list (third (car slot-specs))))
                      (t (print-warning "Invalid hand specified in ALL-FINGERS-TO-HOME request.  No action taken."))))
         (any-to-move nil))
    (when hands
      (dolist (hand hands)
        (dolist (finger '(index middle ring pinkie thumb))
          (let ((home-pos (home-position hand finger))
                (finger-loc (finger-loc-m mtr-mod hand finger)))
            (when (or (not (vpt= home-pos finger-loc))
                      (finger-down hand finger))
              (setf any-to-move t)))))
      (if any-to-move
          (all-home mtr-mod (if (= (length hands) 2) 'both (car hands)))
        (print-warning "ALL-FINGERS-TO-HOME request ignored because fingers are already at home positions.")))))
       
(extend-manual-requests (all-fingers-to-home hand) all-fingers-to-home)
       

;; hold-mouse
;; Like click-mouse except that it does not relase the button.

(defmethod hold-mouse ((mtr-mod dual-execution-motor-module) spec)
  (declare (ignore spec))
  (if (vpt= (loc (right-hand mtr-mod)) #(28 2))
    (hold-punch mtr-mod :hand 'right :finger 'index)
    (model-warning "HOLD-MOUSE requested when hand not at mouse!")))

(extend-manual-requests (hold-mouse) hold-mouse)

;; release-mouse
;; If the mouse button is being held release it.

(defmethod release-mouse-button ((mtr-mod dual-execution-motor-module) spec)
  (declare (ignore spec))
  (if (vpt= (loc (right-hand mtr-mod)) #(28 2))
    (release mtr-mod :hand 'right :finger 'index)
    (model-warning "RELEASE-MOUSE requested when hand not at mouse!")))

(extend-manual-requests (release-mouse) release-mouse-button)


#|
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
|#