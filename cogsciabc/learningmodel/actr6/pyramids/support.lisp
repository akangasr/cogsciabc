;defining metacognitive module

(clear-all)

(undefine-module metacognitive)

(defstruct metacognitive-module delay busy error)

(defun create-metacognitive (model)
  (declare (ignore model))
  (make-metacognitive-module))

(defparameter *flag* nil)
(defparameter *data* nil)

(defun request-metacognitive-chunk (instance buffer-name chunk-spec)
 (if (metacognitive-module-busy instance)
   (model-warning "metacognitive request made to the ~S buffer while the metacognitive module was busy. New request ignored." buffer-name)
 (progn
   (setf (metacognitive-module-busy instance) t)
   (setf (metacognitive-module-error instance) nil)
      (let* ((delay .5) description
           (values (mapcar 'third (chunk-spec-slot-spec chunk-spec)))
             (type (chunk-spec-chunk-type chunk-spec))
           (slots (mapcar 'second (chunk-spec-slot-spec chunk-spec)))
             (nextpos (position 'next slots))
             (contents (if (not nextpos) (mapcan 'list slots values)
                         (mapcan 'list (remove (nth nextpos slots) slots) (remove (nth nextpos values) values))))
             (spec (append '(isa operator) contents)) chunk)
     (cond ((member type '(execute-operator change-operator))           (setf *flag* 1)
            (setf chunk (add-dm-fct (list spec)))     (eval `(sdp ,chunk :base-level -1))
            (schedule-event-relative delay
                                (lambda () (sdp-fct (list chunk :base-level 4))))
           (schedule-event-relative .05
                                #'(lambda () (set-buffer-chunk 'retrieval (car chunk)))))
           ((equal type 'pyramid)
            (setf description (report-knowledge (first values) (second values)))))
        (schedule-event-relative .5 'create-chunk
                 :params (list 'metacognitive description '*metacognitive-busy*))
        (schedule-event-relative delay
                                 'set-metacognitive-free
                                 :module 'metacognitive
                                 :output nil
                                 :priority -1010)
        (schedule-event-relative .1
                                 'goal-style-request
                                 :params (list 'metacognitive chunk-spec)
                                 :destination 'metacognitive
                                 :module 'metacognitive
                                 :output nil)))))

(defun create-chunk (buffer-name chunk-description busy)
  ;  (format t "~%Called CREATE-CHUNK in buffer ~A and with description = ~A~%" buffer-name chunk-description)
  (set busy nil)
  (create-new-buffer-chunk buffer-name chunk-description)
  )

(defun report-knowledge (base height)
  (let ((description
         (if (judge-knowledge base height)
           '(isa classify-problem type retrieve)
           '(isa classify-problem type compute))))
    description))


(defun judge-knowledge (base height)
  (let* ((bn (chunk-to-number base))
         (hn (chunk-to-number height))
         (vn (* (/ (- (* bn 2) (1- hn)) 2) hn))
         (value (number-to-chunk vn))
         (probe (list 'isa 'pyramid 'base base 'height height 'value value)))
    (no-output (sdm-fct probe))))

(defun set-metacognitive-free ()
  (let ((im (get-module metacognitive)))
    (if im
        (progn
          (setf (metacognitive-module-busy im) nil)
          t)
      (print-warning "Call to set-metacognitive-free failed"))))


(defun metacognitive-query (instance buffer-name slot value)
  (declare (ignore slot)) ; the only slot is state
  (case value
    (busy (metacognitive-module-busy instance))
    (free (not (metacognitive-module-busy instance)))
    (error (metacognitive-module-error instance))
    (t (print-warning "Unknown state query ~S to ~S module"
                      value buffer-name)
       nil)))

(defun metacognitive-mod-request (instance buffer mods)
  (if (metacognitive-module-busy instance)
    (model-warning "metacognitive modification request made to the ~S buffer while the metacognitive module was busy. New request ignored." buffer)
        (let ((delay .2))
          (setf (metacognitive-module-busy instance) t)
          (schedule-mod-buffer-chunk buffer mods delay :module 'metacognitive)
          (schedule-event-relative delay 'set-metacognitive-free :module 'metacognitive :priority -1 :output nil))))

;;
(defun metacognitive-reset (instance)
  (setf (metacognitive-module-busy instance) nil)
  (setf (metacognitive-module-error instance) nil)
  (sgp :do-not-harvest metacognitive) )

(define-module-fct 'metacognitive
  '(metacognitive)
  nil
  :version "0.2a"
  :documentation "A module for performing fast metacognitive computations"
  :creation 'create-metacognitive
  :query 'metacognitive-query
  :request 'request-metacognitive-chunk
  :buffer-mod 'metacognitive-mod-request
  :reset (list nil #'metacognitive-reset) )

;define all 128 problems
(setf problems '(
(	4	3	9	REGULAR	VALUE	)
(	5	3	12	REGULAR	VALUE	)
(	5	4	14	REGULAR	VALUE	)
(	6	3	15	REGULAR	VALUE	)
(	6	4	18	REGULAR	VALUE	)
(	6	5	20	REGULAR	VALUE	)
(	7	3	18	REGULAR	VALUE	)
(	7	4	22	REGULAR	VALUE	)
(	7	5	25	REGULAR	VALUE	)
(	8	3	21	REGULAR	VALUE	)
(	8	4	26	REGULAR	VALUE	)
(	8	5	30	REGULAR	VALUE	)
(	9	3	24	REGULAR	VALUE	)
(	9	4	30	REGULAR	VALUE	)
(	9	5	35	REGULAR	VALUE	)
(	10	3	27	REGULAR	VALUE	)
(	10	4	34	REGULAR	VALUE	)
(	10	5	40	REGULAR	VALUE	)
(	11	3	30	REGULAR	VALUE	)
(	11	4	38	REGULAR	VALUE	)
(	11	5	45	REGULAR	VALUE	)))

 (defparameter *compiled-actor* '(p* start-output3
   =GOAL>
       ISA ENCODE-TASK
       NEXT-STATE =POST
       STEP ATTENDING
       ROLE =ROLE
   =VISUAL>
       ISA TEXT
       VALUE "9$5"
    =imaginal>
      isa pyramid
    ==>
    +visual>
      isa move-attention
      screen-pos base-position
  !bind! =val (number-to-chunk 35)
  ;!eval! (princ =val)
  !eval! (log-experiment-event 3)
  !eval! (log-experiment-event 4)
  !eval! (log-experiment-event 5)
    +imaginal>
        value =val
    =goal>
       step ready
       state =post
    +manual>
       ISA         press-key
       key         "N"
    +GOAL>
       isa respond-task
       STATE ANSWER
       STEP START
       VALUE =val
       pos 0
       type no-check))

(defparameter *efficient-looker* '(p look-base-loc1
     =goal>
      isa encode-task
      state encode-base
      step ready
    =imaginal>
      isa pyramid
    ?retrieval>
      state free
    ?imaginal>
      state free
  ==>

     +visual-location>
        isa visual-location
        screen-x lowest
        :attended nil
     +goal>
      next-state encode-height
      step looking
      role base
    +imaginal> =imaginal
  ))

;global parameters
(defparameter *position* 150)
(defparameter *problem* nil)
(defparameter *problem-result* nil)
(defparameter *problem-start-time* nil)
(defparameter *in-nback* nil)
(defparameter *rd-letter* nil)
(defparameter *timeout-event* nil)
(defparameter *experiment-events* nil)
(defparameter *nback-match* 0)
(defparameter *finger-count* nil)

;; JJ: log data
(defvar *log* nil)
(defvar *rt* -2.6)
(defvar *lf* 0.1)
(defvar *blc* 0.0)
(defvar *ans* nil)
(defvar *out* t)

;; A helper for returning from the *log* only relevant entries.
(defun get-log-state-mean (log state height nth)
  (let ((all
         (loop for l in log
            when (and (= (elt l 0) state)
                      (= (elt l 1) height))
            collect (elt l (+ 2 nth)))))
    (/ (apply #'+ all) (length all))))

(defun run-problems (problems)
  (format t "Running pyramids with rt = ~a, lf = ~a, blc = ~a, and = ~a ~%"
          (first ext:*args*) (second ext:*args*) (third ext:*args*) (fourth ext:*args*))
  (setf *rt* (read-from-string (first ext:*args*)))
  (setf *lf* (read-from-string (second ext:*args*)))
  (when (third ext:*args*)
    (setf *blc* (read-from-string (third ext:*args*))))
  (when (fourth ext:*args*)
    (setf *ans* (read-from-string (fourth ext:*args*))))
  (mapcar #'(lambda (x) (run-problem-state1 x t) (module-engagement t)) problems)
  (mapcar #'(lambda (x) (run-problem-state2 x t) (module-engagement t)) problems)
  (mapcar #'(lambda (x) (run-problem-state3 x t)(module-engagement t)) problems)
  (with-open-file (out "pyramids.csv" :direction :output :if-exists :supersede)
    (format *out* "RESULT_OUT stage,height,response,val~%")
    (dolist (s (list 1 2 3))
      (dolist (h (list 3 4 5))
        (format *out* "RESULT_OUT ~a,~a,\"encode\",~a~%" s h (get-log-state-mean *log* s h 0))
        (format *out* "RESULT_OUT ~a,~a,\"solve\",~a~%" s h (get-log-state-mean *log* s h 1))
        (format *out* "RESULT_OUT ~a,~a,\"respond\,~a~%" s h (get-log-state-mean *log* s h 2)))))
  (when (third ext:*args*) (close *out*)))

(defun run-problem-state1 (problem fstream &optional (flag nil) (boldflag nil))
  (let ((base (first problem)) (height (second problem)) temp (cond 1)
        (value (third problem)))
    (setf *finger-count* nil)
    ;(format t " ~A ~A ~A ~A " cond base height value)
    (setf *problem* (list base '$ height '= 'x base height value))
    (reset)
    (eval `(sgp :rt ,*rt*))
    (eval `(sgp :lf ,*lf*))
    (eval `(sgp :blc ,*blc*))
    (eval `(sgp :ans ,*ans*))

    (eval `(sgp :v ,flag))
    ;; JJ: new params set.    
    (let* ((window (open-exp-window "Pyramid Experiment" :visible nil))
           (items (list (first *problem*)   (third *problem*) ))
           (wait-time nil)
           (model-output nil))
        (setf *experiment-events* nil)
        (log-experiment-event 1)
        (setf wait-time (synchronize-fixation))
        (schedule-event-relative 0
          (lambda ()
            (setf *problem-start-time* (/ (get-time) 1000.0))
            (mod-focus-fct (list 'state 'start-up 'step 'ready 'next-state nil 'role nil 'value nil))
            (clear-buffer 'imaginal)
            (clear-buffer 'retrieval)
            (setf *position* 150)
            (install-device window)))
        (schedule-event-relative 2
           (lambda ()
             (log-experiment-event 2)
             (present-problem items)
             (proc-display :clear t)))
        (setf *timeout-event* (schedule-event-relative (+ wait-time 30) 'schedule-post-answer))
        (run 32)
        (if (< (length *experiment-events*) 6)
          (terpri)
          (progn
            (run 30)
            (sgp :v t)
            (log-experiment-event 8)
            (if (= (length *experiment-events*) 8)
              (progn
                (setf temp (reverse (mapcar #'(lambda (x y) (- (car x) (car y))) (reverse (cdr (reverse *experiment-events*))) (cdr *experiment-events*))))
                ;;(format t "lal ~6,2F lal ~6,2F lal ~6,2F" (second temp) (fourth temp) (fifth temp))
                (push (list 1 height (second temp) (fourth temp) (fifth temp))
                      *log*)
                ;;(format t "~a ~a ~a ~a" height (second temp) (fourth temp) (fifth temp))
                (if boldflag (predict-bold-data fstream))
                ))
            ))
        )))

(defun run-problem-state2 (problem fstream &optional (flag nil) (boldflag nil))
  (let ((base (first problem)) (height (second problem)) temp (cond 2)
        (value (third problem)))
    (setf *finger-count* nil)
    (setf *problem* (list base '$ height '= 'x base height value))
    (reset)
    (eval `(add-dm (solution isa pyramid base ,(number-to-chunk base) height ,(number-to-chunk height) value ,(number-to-chunk value))))
    (sdp solution :base-level -2.25)
    (eval `(sgp :v ,flag))
    (eval `(sgp :rt ,*rt*))
    (eval `(sgp :lf ,*lf*))
    (let* ((window (open-exp-window "Pyramid Experiment" :visible nil))
           (items (list (first *problem*)   (third *problem*) ))
           (wait-time nil))
        (setf *experiment-events* nil)
        (log-experiment-event 1)
        (setf wait-time (synchronize-fixation))
         (eval *efficient-looker*)
      (spp LOOK-BASE-LOC1 :u 1)
        (schedule-event-relative 0
          (lambda ()
            (setf *problem-start-time* (/ (get-time) 1000.0))
            (mod-focus-fct (list 'state 'start-up 'step 'ready 'next-state nil 'role nil 'value nil))
            (clear-buffer 'imaginal)
            (clear-buffer 'retrieval)
            (setf *position* 150)
            (install-device window)))
        (schedule-event-relative 2
           (lambda ()
             (log-experiment-event 2)
             (present-problem items)
             (proc-display :clear t)))
        (setf *timeout-event* (schedule-event-relative (+ wait-time 30) 'schedule-post-answer))
        (run 32)
        (if (< (length *experiment-events*) 6)
          (terpri)
          (progn
            (run 30)
            (sgp :v t)
            (log-experiment-event 8)
            (if (= (length *experiment-events*) 8)
              (progn
                (setf temp (reverse (mapcar #'(lambda (x y) (- (car x) (car y))) (reverse (cdr (reverse *experiment-events*))) (cdr *experiment-events*))))
                (push (list 2 height (second temp) (fourth temp) (fifth temp))
                      *log*)
                (if boldflag (predict-bold-data fstream))))
            )))))

(defun run-problem-state3 (problem fstream &optional (flag nil) (boldflag nil))
  (let ((base (first problem)) (height (second problem)) temp (cond 2)
        (value (third problem))
        (production (subst (problem-to-string problem) "9$5" (subst (third problem) 35 *COMPILED-ACTOR*) :test 'equal)))
    (setf *finger-count* nil)
    (setf *problem* (list base '$ height '= 'x base height value))
    (reset)
    (eval `(add-dm (solution isa pyramid base ,(number-to-chunk base) height ,(number-to-chunk height) value ,(number-to-chunk value))))
    (eval `(sgp :v ,flag))
    (eval `(sgp :rt ,*rt*))
    (eval `(sgp :lf ,*lf*))
    (eval production)
    (spp start-output3 :u 1)
    (let* ((window (open-exp-window "Pyramid Experiment" :visible nil))
           (items (list (first *problem*)   (third *problem*) ))
           (wait-time nil))
        (setf *experiment-events* nil)
        (log-experiment-event 1)
        (setf wait-time (synchronize-fixation))
        (schedule-event-relative 0
          (lambda ()
            (setf *problem-start-time* (/ (get-time) 1000.0))
            (mod-focus-fct (list 'state 'start-up 'step 'ready 'next-state nil 'role nil 'value nil))
            (clear-buffer 'imaginal)
            (clear-buffer 'retrieval)
            (setf *position* 150)
            (install-device window)))
        (schedule-event-relative 2
           (lambda ()
             (log-experiment-event 2)
             (present-problem3 items)
             (proc-display :clear t)))
        (setf *timeout-event* (schedule-event-relative (+ wait-time 30) 'schedule-post-answer))
        (run 32)
        (if (< (length *experiment-events*) 6)
          nil
          ;(terpri)
          (progn
            (run 30)
            ;(terpri)
            (sgp :v t)
            (log-experiment-event 8)
            (if (= (length *experiment-events*) 8)
              (progn
                (setf temp (reverse (mapcar #'(lambda (x y) (- (car x) (car y))) (reverse (cdr (reverse *experiment-events*))) (cdr *experiment-events*))))
                (push (list 3 height (second temp) (fourth temp) (fifth temp))
                      *log*)
                (if boldflag (predict-bold-data fstream))))
              ;(terpri)
            )))))

;functions below for implementing experiment
(defmethod rpm-window-key-event-handler ((win rpm-window) key)
  (if (not *in-nback*)
    (if (not (string-equal key "r"))
      (if (not (string-equal key "c"))
        (progn
        (add-text-to-exp-window :text (string key) :x *position* :y 90 :width 25)
        (proc-display)
        (push (list (- (/ (get-time) 1000.0) *problem-start-time*) key) *problem-result*)
        (setf *position* (+ *position* 8)))
        )
      (progn
        (delete-event *timeout-event*)
        (schedule-post-answer))))
  )

(defmethod device-hold-finger ((win rpm-window) hand finger)
 (declare (ignore hand finger)))
(defmethod device-release-finger ((win rpm-window) hand finger)
 (declare (ignore hand finger)))

(defun synchronize-fixation ()
  ; scanner waits for two pulses after displaying cross
  (let* ((current-scan (/ (get-time) 2000.0))
        (next-scan-start (ceiling current-scan))
        (diff (- next-scan-start current-scan))
        (wait-time (+ 2 (* 2 diff))))
    wait-time))

(defun log-experiment-event (val)
  (push (list (/ (get-time) 1000.0) val) *experiment-events*))

(defun present-problem (items)
  (add-text-to-exp-window :text (format nil "~A" (first items)) :x 75 :y 60 :width 25)
  (add-text-to-exp-window :text "$" :x 100 :y 60 :width 25)
  (add-text-to-exp-window :text (format nil "~A" (second items)) :x 125 :y 60 :width 25))


(defun present-problem3 (items)
           (add-text-to-exp-window :text (problem-to-string items) :x 100 :y 60 :width 25))




(defun schedule-post-answer ()
  (let ((ncount (+ 3 (random 6))))
  ; T+0: feedback
  ; T+5: first rd letter
  ; T+6: first letter gone
  ; T+6.25: second letter
  ; T+7.25: second letter gone
  ; T+7.5: 3rd letter
  ; ...
  ; T+16.25: 10th letter appears
  ; T+17: rd over
  (schedule-event-relative 0 (lambda () ; feedback start
        (log-experiment-event 6)
        (setf *problem-result* (reverse *problem-result*))
        (let ()
          (clear-exp-window)
          (present-feedback *problem*))
          (proc-display :clear t)))
  (schedule-event-relative 4 (lambda () ; after feedback
        (setf *in-nback* t)
        (log-experiment-event 7)
        (clear-exp-window)
        (present-repetition-detection)))
  (loop for i from 1 to ncount do
        (schedule-event-relative (+ 4 (* 1.25 (- i 1))) (lambda ()
                                                        (present-rd-number i)
                                                        (proc-display :clear t)))
        (schedule-event-relative (+ 5 (* 1.25 (- i 1))) (lambda ()
                                                        (clear-rd-number)
                                                        (proc-display :clear t))))
  (schedule-event-relative (+ 4 (* ncount 1.25)) (lambda ()
                                (setf *in-nback* nil)))
  (schedule-break-relative (+ 4.001 (* ncount 1.25)))
  nil))

(defun present-feedback (problem)
 (cond ((atom (sixth problem))
         (add-text-to-exp-window :text
                                 (format nil "~A" (sixth problem)) :x 75 :y 60 :width 25 :color 'green))
        (t (add-text-to-exp-window :text
                                   (format nil "~A" (first (sixth problem))) :x 50 :y 60 :width 25 :color 'green))
           (add-text-to-exp-window :text
                                   (format nil "~A" (second (sixth problem)) :x 75 :y 60 :width 25 :color 'green)))
  (add-text-to-exp-window :text "$" :x 100 :y 60 :width 25  :color 'green)
  (add-text-to-exp-window :text (format nil "~A" (seventh problem)) :x 125 :y 60 :width 25  :color 'green)
  (add-text-to-exp-window :text "=" :x 150 :y 60 :width 25  :color 'green)
  (cond ((atom (eighth problem))
         (add-text-to-exp-window :text (format nil "~A" (eighth problem)) :x 175 :y 60 :width 25   :color 'green))
        (t (add-text-to-exp-window :text (concatenate 'string (format nil "~A" (first (eighth problem))) "_"
                                                      (format nil "~A" (second (eighth problem)))) :x 175 :y 60 :width 50 :color 'green))))

(defun present-repetition-detection ()
  (add-button-to-exp-window :x 80 :y 155 :height 60 :width 120 :text "Match!"))

(defun present-rd-number (num)
  (if (and (> num 1)
           (< (act-r-random 1.0) .25))
    (setf *nback-match* 1)
    (setf *nback-match* 0))
  (setf *rd-letter* (add-text-to-exp-window :text (format nil "~A" num) :x 75 :y 100 :width 25)))

(defun clear-rd-number ()
  (remove-items-from-exp-window *rd-letter*))

(defun reset-nback () (setf *nback-match* -1))

;finger counting
(defun next-finger ()
  (case *finger-count*
    (1 'index)
    (2 'middle)
    (3 'ring)
    (4 'pinkie)
    (5 'thumb)
    (6 'index)
    (7 'middle)
    (8 'ring)
    (9 'pinkie)))

(defun next-number (number)
  (case number
    (n1 'n2)
    (n2 'n3)
    (n3 'n4)
    (n4 'n5)
    (n5 'n6)
    (n6 'n7)
    (n7 'n8)
    (n8 'n9)
    (n9 'n10)))


; works on number chunks
(defun chunk-to-number (x)
  (read-from-string (subseq (symbol-name x) 1)))

; only works on number chunks
(defun number-to-chunk (x)
  (intern (concatenate 'string "N" (format nil "~A" x))))

(defun problem-to-string (lis)
  (concatenate 'string (format nil "~A" (first lis)) "$" (format nil "~A" (second lis))))

(defun chunk-to-string (x)
  (format nil "~A" (chunk-to-number x)))

(defun string-to-chunk (x)
  (let ((name  (number-to-chunk (read-from-string x)))
        (val (read-from-string x)))
    (cond ((and (numberp val) (null (no-output (eval `(dm ,name)))))  (car (add-dm-fct `((,name isa equation-token string ,x))))
           (eval `(sdp ,name :base-level 2)) name)
          (t name))))

(defun dereference (string)
  (number-to-chunk (read-from-string string)))

;functions for accessing numbers called in model
(defun compose-fraction (n1 n2)
  (string-to-chunk (concatenate 'string (eval `(chunk-slot-value ,n1 string)) "/" (eval `(chunk-slot-value ,n2 string)))))

(defun compose-number (n1 n2)
  (string-to-chunk (concatenate 'string (eval `(chunk-slot-value ,n1 string))(eval `(chunk-slot-value ,n2 string)))))

(defun classify (term)
  (cond ((eq term 'x) 'variable)
        (t (let ((val (chunk-to-number term)))
             (cond ((< val 0) 'negative)
                   ((> val 99) 'big)
                   ((and (numberp val) (rationalp val) (not (integerp val))) 'fraction)
                   ((<  val 100) 'digit))))))

(defun mod-tens (n1)
  (let ((string   (eval`(chunk-slot-value ,n1 string))))
    (string-to-chunk (string (aref string (1- (length string)))))))

(defun hundredsp (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
    (and (numberp x) (> x 99))))

(defun positive-digitp (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
    (and (numberp x) (integerp x) (< x 10) (> x 0))))

(defun digitp (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
    (and (numberp x) (integerp x) (< (abs x) 10))))

(defun hundreds-close (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
    (and (numberp x) (integerp x) (< (mod x 100) 20))))

(defun fivep (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
    (and (numberp x) (integerp x) (> (mod x 10) 5))))

(defun switch-sign (chunk)
  (let ((x (chunk-to-number chunk)))
    (number-to-chunk (- x))))

(defun multi-digitp (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
    (and (numberp x) (integerp x) (> x 9) )))

(defun fractionp (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
     (and (numberp x) (rationalp x) (not (integerp x)))))

(defun integer-part (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
     (number-to-chunk (floor x))))


(defun negativep (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
     (and (numberp x) (< x 0))))

(defun bigp (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
     (and (numberp x) (>= x 100))))

(defun smallp (chunk)
  (let ((x (if (not (eq chunk 'x)) (chunk-to-number chunk))))
     (and (numberp x) (< x 50))))

(defun extract-tens (num) (number-to-chunk (floor (/ (read-from-string (eval `(chunk-slot-value ,num string))) 10))))

(defun extract-ones (num) (number-to-chunk  (mod (read-from-string (eval `(chunk-slot-value ,num string))) 10)))

(defun make-hundreds (num)
  (full-number-chunk (* 100 (read-from-string (eval `(chunk-slot-value ,num string))))))

(defun full-number-chunk (value)
  (let* ((chunk (number-to-chunk value))
        (string (format nil "~A" value)))
    (if (null (no-output (eval `(dm ,chunk))))
      (car (add-dm-fct `((,chunk isa equation-token string ,string)))) chunk)))

(defun increment-hundreds (num) (full-number-chunk  (+ 1 (read-from-string (eval `(chunk-slot-value ,num string))))))

(defun hundreds-round (num)
  (let* ((value (floor (read-from-string (eval `(chunk-slot-value ,num string))))))
    (number-to-chunk (round value 100))))

(defun extract-denominator (num)
  (let* ((value (read-from-string (eval `(chunk-slot-value ,num string))))
         (denominator (round (/ 1 (- value (floor value))))))
    (number-to-chunk denominator)))

(defun extract-numerator (num)
  (let* ((value (read-from-string (eval `(chunk-slot-value ,num string))))
         (denominator (round (/ 1 (- value (floor value)))))
         (numerator (* value denominator)))
    (number-to-chunk numerator)))


;functions for creating initial data base
(defun ones (x) (number-to-chunk (mod (chunk-to-number x) 10)))
(defun tens (x) (number-to-chunk (floor (/ (chunk-to-number x) 10))))
(defun increment (x) (number-to-chunk (+ (chunk-to-number x) 1)))
(defun add-up (x y) (number-to-chunk (+ (chunk-to-number x) (chunk-to-number y))))
(defun decrement (x) (number-to-chunk (+ (chunk-to-number x) -1)))
(defun add (x y) (number-to-chunk (+ (chunk-to-number x) (chunk-to-number y))))
(defun tostring (x) (number-to-chunk x))


(defparameter *min-dm-number* -20)
(defparameter *max-dm-number* 99)
(defun add-numbers-to-dm ()
  (loop for i from *min-dm-number* to *max-dm-number* do
        (let ((string-val (format nil "~A" i))
               (ones (number-to-chunk (mod i 10)))
               (tens (number-to-chunk (floor (/ i 10)))))
         (if (> i 9) (add-dm-fct (list (list (intern (concatenate 'string "N" string-val)) 'isa 'equation-token 'string string-val 'ones ones 'tens tens)))
           (add-dm-fct (list (list (intern (concatenate 'string "N" string-val)) 'isa 'equation-token 'string string-val)))))))

(defun addition-base-level (a b)
  (let ((fact (car (eval `(no-output (sdm isa fact arg1 ,(number-to-chunk a) arg2 ,(number-to-chunk b) relation +))))))
    (eval `(sdp ,fact :base-level ,(- 3.5 (log (* a b)))))))

(defun generate-+-fact (term1 term2)
  (list 'isa 'fact 'arg1 (number-to-chunk term1) 'arg2 (number-to-chunk term2) 'arg3 (number-to-chunk (+ term1 term2))
        'relation '+))

(defun generate-minus-fact (term1 term2)
  (list 'isa 'fact 'arg1 (number-to-chunk (+ term1 term2)) 'arg2 (number-to-chunk term2) 'arg3 (number-to-chunk term1)
        'relation '-))

(defun generate-*-fact (term1 term2)
  (list 'isa 'fact 'arg1 (number-to-chunk term1) 'arg2 (number-to-chunk term2) 'arg3 (number-to-chunk (* term1 term2))
        'relation '*))

(defun generate-div-fact (term1 term2)
  (list 'isa 'fact 'arg1 (number-to-chunk (* term1 term2)) 'arg2 (number-to-chunk term2) 'arg3 (number-to-chunk term1)
        'relation '/))

(defun gen-facts ()
  (add-dm-fct (mapcan #'(lambda (x) (mapcan #'(lambda (y) (list (generate-+-fact x y)(generate-*-fact x y)
                                                                (generate-div-fact x y)(generate-minus-fact x y)))
                                            '(-1 0 1 2 3 4 5 6 7 8 9 10)))
                      '(-1 0 1 2 3 4 5 6 7 8 9 10))))

(defun ballpark-division (a b)
  (list 'isa 'fact 'arg1 (number-to-chunk a) 'arg2 (number-to-chunk b) 'arg3 (number-to-chunk (floor (/ a b)))
        'relation 'ball/))

(defun ballpark-facts ()
  (do ((i 1 (1+ i)))
      ((= i 10) nil)
    (do ((j 0 (1+ j))
          (result nil (cons (ballpark-division j i) result)))
         ((= j (* i 10)) (mapcar #'(lambda (x) (sdp-fct (list x :base-level -1.5))) (add-dm-fct result))))))

;code for module reporting
(defun module-engagement (fstream)
  (let* ((bm (get-module bold))
         (data  (reverse (parse-and-augment-trace-lists-for-bold bm)))
         (result  (mapcan #'(lambda (x) (extract-durations x *experiment-events*)) data)))
    ;; JJ: removed this, no needed to output module data
    ;;(mapcar #'(lambda (x) (format fstream "~6,3F" x)) result)
    )
    (terpri))

(defun parse-and-augment-trace-lists-for-bold (bm)
   (let* ((trace (get-current-buffer-trace))
          (b (bold-module-buffers bm))
          (buffers (if (listp b) b (buffers)))
          (all-data nil))
     (dolist (x buffers) ; for each traced buffer
       (let ((rects nil)
             (current-rect nil)
             (current-request nil))
           (dolist (z trace) ; for each time in the trace
             (let* ((record (find x (buffer-record-buffers z) :key 'buffer-summary-name))
                   (request (buffer-summary-request record))) ; record is the record structure for the buffer at the time
               (if current-rect ; if there is already an event going on
                  (when (or (null (buffer-summary-busy record)) ; and it should end
                             (buffer-summary-busy->free record)
                             (buffer-summary-request record))
                    (push (cons current-rect (list (buffer-record-time-stamp z) current-request)) rects)  ; add the start->end event
                    (if (buffer-summary-request record) ; and start a new event if necessary
                         (progn (setf current-rect (buffer-record-time-stamp z)) (setf current-request request))
                       (progn (setf current-rect nil) (setf current-request nil))))
                ; current-rect false
                 (if (buffer-summary-busy record) ; if busy,
                   (if (and (buffer-summary-request record) ; if there is a request and a chunk name, or error is set and error->clear is not, or busy->free is set
                            (or (buffer-summary-chunk-name record)
                                (and (buffer-summary-error record)
                                     (not (buffer-summary-error->clear record)))
                                (buffer-summary-busy->free record)))
                       (push (cons (buffer-record-time-stamp z) (list (buffer-record-time-stamp z) request)) rects) ; add a record starting and ending at this time
                     (progn (setf current-rect (buffer-record-time-stamp z)) (setf current-request request))) ; otherwise save the time stamp in current-rect
                   (if (buffer-summary-request record) ; if not busy, but there is a request,
                       (push (cons (buffer-record-time-stamp z) (list (buffer-record-time-stamp z) request)) rects) ; add a record starting and ending at this time
                     (when (buffer-summary-chunk-name record) ; if not busy and not request, but chunk-name is set, do the same
                       (push (cons (buffer-record-time-stamp z) (list (buffer-record-time-stamp z) request)) rects)))))))
         (push (cons x (reverse rects)) all-data)))
      all-data))

(defun extract-durations (module events)
  (setf module (cdr module))
  (let* ((events (mapcar 'reverse events))
         (boundaries (mapcar #'(lambda (x) (second (assoc x events))) '(2 3 5 6))))
    (do ((temp boundaries (cdr temp))
         (result nil (cons (engagement module (car temp) (second temp)) result)))
        ((null (cdr temp)) (reverse result)))))

(defun engagement (module start stop)
  (do ((temp module (cdr temp))
       (sum 0 (cond ((< (second (car temp)) start) sum)
                    ((< (first (car temp)) start) (+ sum (- (second (car temp)) start)))
                    ((> (second (car temp)) stop) (+ sum (- stop (first (car temp)))))
                    (t (+ sum (- (second (car temp)) (first (car temp))))))))
      ((or (null temp) (> (first (car temp)) stop))
       (if (> stop start) (/ sum (- stop start)) 0))))

(defun predict-bold-data (fstream &optional (start 0) (end (+ 11 (mp-time))))
  (let* ((bm (get-module bold))
         (data  (parse-and-augment-trace-lists-for-bold bm))
         (bold nil)
         (point (bold-module-point bm))
         (inc (bold-module-inc bm)))
    ;(setf data (append (list (car data) (second data)
     ;                         (cons (car (third data)) (pepper-peaks (cdr (third data)))))
     ;                   (cdddr data)))
    (if (< (- end start) inc)
        (print-warning "Sample time too short for BOLD predictions - must be at least :bold-inc seconds (currently ~s)" inc)
      (progn
        (unless (zerop (mod start inc))
          (setf start (* inc (floor start inc)))
          (model-warning "Start time should be a multiple of :bold-inc (~S).  Using start time of ~S."
                         inc start))

        (dolist (x data)
          (if (find (car x) point)
              (push (cons (car x) (bold_point-predict (mapcar #'car (cdr x)) bm start end)) bold)
            (push (cons (car x) (bold_interval-predict (cdr x) bm start end)) bold)))

        ;; Cache the max values for each buffer so that they can
        ;; be used in normalizing things later if desired

        ; This can cause errors with long runs, too many args to max
      ; (dolist (x bold)
      ;   (let* ((buffer (car x))
      ;          (data (cdr x))
      ;          (max (when data (apply #'max data))))
      ;     (when (or (null (gethash buffer (bold-module-max-table bm)))
      ;               (> max (gethash buffer (bold-module-max-table bm))))
      ;       (setf (gethash buffer (bold-module-max-table bm)) max))))
       ; (output-bold-response-data bold bm start end)
        ;(setf xx bold)
        (output-jra (mapcar 'add-tail bold) fstream data)))))

(defun output-jra (bold fstream data)
(setf *data* data)
  (let* ((durations (module-durations data *experiment-events*))
         (max (car (array-dimensions durations)))
        (occupancy (state-occupancy *experiment-events*)))
    (do ((temp bold (mapcar 'cdr temp))
         (i 1 (1+ i)))
        ((null (car temp)) nil)
      (format fstream "~3D" i)
      (do ((temp1 (mapcar 'car temp) (cdr temp1)))
          ((null temp1) nil)
        (format fstream "~8,3F"     (car temp1)))
      (do ((j 0 (1+ j)))
           ((= j 7) (format fstream "~%"))
           (format fstream "~7,3F"
                   (if (< i max) (aref occupancy (1- i) j)
                     (if (= j 6) 1 0)))))))

(defun state-occupancy (lis)
  (let* ((todo  (mapcar #'(lambda (x) (+ (car x) 2)) (cddr (reverse lis))))
        (current 4)
        (state 1)
        (used 0)
        (index 2)
        (length (1+ (round (/ (caar lis) 2))))
        (scans (make-array (list length 7) :initial-element 0.0)))
    (setf (aref scans 0 0) 1) (setf (aref scans 1 0) 1)
   (loop (cond ((or (null todo) (= index length)) (return scans))
                ((> (car todo) (+ 2 current))
                 (setf (aref scans index state) (/ (- 2 used) 2.0))
                 (incf index) (setf used 0) (setf current (+ 2 current)))
                (t (setf (aref scans index state) (/ (- (- (car todo) current) used) 2))
                   (incf state) (setf used (- (car todo) current)) (setf todo (cdr todo)))))))

(defun module-durations (data lis)
  (setf data (reverse data))
  (let* ((length (1+ (round (/ (caar lis) 2))))
          (scans (make-array (list length 6) :initial-element 0.0)))
         (do ((i 0 (1+ i)))
             ((= i 6) scans)
           (let ((todo  (cdr (nth i data)))
                 (current 0)
                 (index 1))
             (loop (cond ((or (null todo) (= index length)) (return scans))
                         ((> (first (car todo)) (+ 2 current))
                          (incf index) (setf current (+ 2 current)))
                         ((> (second (car todo)) (+ 2 current))
                          (setf current (+ 2 current))
                          (setf (aref scans index i) (+ (aref scans index i) (- current (caar todo))))
                          (incf index) (setf todo (cons (cons current (cdr (car todo))) (cdr todo))))
                         ((= (first (car todo))(second (car todo)))
                          (setf (aref scans index i) (+ (aref scans index i) .05))(setf todo (cdr todo)))
                          (t (setf (aref scans index i) (+ (aref scans index i) (- (second (car todo)) (first (car todo)))))
                             (setf todo (cdr todo)))))))))


(defun add-tail (bold)
  (let ((length (length bold)))
    (do ((tail (nthcdr (- length 4) bold) (cdr tail))
         (old (cdr bold) (cdr old))
         (new (list (nth (- length 5) bold)) (cons (+ (car tail) (car old)) new)))
        ((null tail) (append (reverse new) (reverse (nthcdr 5 (reverse old))))))))
