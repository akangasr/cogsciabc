(clear-all)

(define-model pyramids
  (sgp :esc t :bll nil :ol 1 :egs .01 :ans  nil :er t :TRACE-DETAIL low  :MODEL-WARNINGS nil :v nil :RANDOMIZE-TIME nil  :imaginal-delay .1)
  ;(sgp :MIN-FITTS-TIME .15 :MOTOR-BURST-TIME .075 :MOTOR-FEATURE-PREP-TIME .075 :MOTOR-INITIATION-TIME .075 :PECK-FITTS-COEFF .1125)
  (sgp :MIN-FITTS-TIME .15 :MOTOR-BURST-TIME .075 :MOTOR-INITIATION-TIME .075 :PECK-FITTS-COEFF .1125)
  (sgp :save-buffer-trace t :traced-buffers (visual imaginal retrieval manual metacognitive) :bold-inc 2 :bold-exp 5 :bold-scale 1 :neg-bold-exp 15 :neg-bold-scale 1 :bold-positive 6 :bold-negative 1)
  (set-visloc-default isa visual-location screen-x lowest screen-y lowest :attended new)
  (add-word-characters #\- #\/) 

  (chunk-type definition task)
  (chunk-type encode-task state step next-state role  value)
  (chunk-type respond-task state step value pos type)
  (chunk-type feedback-task state step next-state role  value)
  (chunk-type pyramid base height value sum term result extra)
  (chunk-type operator pre action arg1 arg2 result post  fail success )
   (chunk-type execute-operator pre action arg1 arg2 result   fail success  post next)
  (chunk-type instruction-task state step arg1 arg2 arg3 relation result post destination problem)  
  (chunk-type fact arg1 arg2 arg3 relation)
  (chunk-type equation-token string  ones tens)
  (chunk-type item-location item location)
   (chunk-type addition-task step parent)
   (chunk-type multiplication-task step parent)
   (chunk-type addition-problem number1 number2 tens1 tens2 ones1 ones2 onesa tensa carry answer)
    (chunk-type multiplication-problem number1 number2 tens1 tens2 ones1 ones2 onesa tensa carry answer)
   (chunk-type feedback base height value status)
  (chunk-type key-location)
  (chunk-type classify-problem base height value type)

 (add-word-characters #\$)
  (add-numbers-to-dm)

  (add-dm 
   (do-pyramid isa definition task pyramid)
    (goal isa encode-task state nil step nil next-state nil role nil value nil)
    (x isa equation-token string "x" )
    (dollar-sign isa equation-token string "$")
    (equal-sign isa equation-token string "=")
    (plus-sign isa equation-token string "+")
    (minus-sign isa equation-token string "-")
    (base-position isa visual-location screen-x  75   screen-y  60)
    (height-position isa visual-location screen-x  125   screen-y  60)
    (value-position isa visual-location screen-x  175   screen-y  60)
    (base-il isa item-location item base location base-position)
    (height-il isa item-location item height location height-position)
    (value-il isa item-location item value location value-position)

(pyramid1
  ISA OPERATOR
   PRE  start-pyramid
   ACTION  variable-set
   ARG1  sum
   arg2  base
   POST  sum-initialized)
(pyramid2
  ISA OPERATOR
   PRE  sum-initialized
   ACTION  subtract
   ARG1  base
   arg2 n1
   result term
   POST  term-set)
(pyramid3
  ISA OPERATOR
   PRE  term-set
   ACTION  count-set
   ARG1  n2
   POST  count-set)
(pyramid4
  ISA OPERATOR
   PRE  count-set
   ACTION  add
   ARG1  term
   arg2 sum
   result sum
   POST  sum-set)
(pyramid5
  ISA OPERATOR
   PRE  sum-set
   ACTION  test-equal
   ARG1  count
   ARG2  height
   success  respond
   fail iterate)
(pyramid6
  ISA OPERATOR
   PRE  respond
   ACTION  output
   arg1 sum
   post attend-feedback)
(pyramid7
  ISA OPERATOR
   PRE  iterate
   action decrement-count
   action decrement
   arg1 term
   post decrement-done)
(pyramid8
   isa operator
   pre decrement-done
   ACTION  count-increment
   post count-set)
)





  (gen-facts)
  (sdp :base-level *blc*)


  (mapcar #'(lambda (x) (mapcar #'(lambda (y) (addition-base-level x y))
                                            '(1 2 3 4 5 6 7 8 9)))
                      '(1 2 3 4 5 6 7 8 9))





  (ballpark-facts)

  (p start-up
     =goal>
      isa encode-task
      state start-up
      step ready
    ?retrieval>
      state free
      buffer empty
  ==>
    +manual>
       ISA         press-key
       key         "C"
    +goal>
        step hand-ready
  )

(p start-up1
     =goal>
      isa encode-task
      state start-up
      step hand-ready
    ?manual>
      state free
  ==>
    +retrieval>
      isa definition
    =goal>
       step get-task
  )

(p start-up2
     =goal>
      isa encode-task
      state start-up
      step get-task
    =retrieval>
       isa definition
       task pyramid
  ==>
    =goal>
       state encode-base
       step ready
    +imaginal>
      isa pyramid
  )

;productions that encode
   (p look-base-loc 
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
      next-state encode-dollars
      step looking  
      role base
    +imaginal> =imaginal
  )

   (p look-dollars-loc 
     =goal>
      isa encode-task
      state encode-dollars
      step ready
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
      role skip
  )

  (p lookup-height-loc
     =goal>
      isa encode-task
      state encode-height
      step ready
    ?retrieval>
      state free
      buffer empty
  ==>
     +visual-location>
        isa visual-location
        screen-x highest
        :attended nil
     +goal>
      next-state determine-answer
      step looking
      role height
  )




  (p retrieve-location
    =goal>
        isa encode-task
        step retrieving
    =retrieval>
      isa item-location
      location =loc
  ==>
    +goal>
        step looking
    +visual-location>
        isa visual-location
        :nearest =loc
  )
  
(p look
    =goal>
      isa encode-task
      step looking
    =visual-location>
      isa visual-location
    =imaginal>
      isa pyramid
  ?manual>
    state free
  ?imaginal>
       state free
    ==>
    +visual>
      isa move-attention
      screen-pos =visual-location
    +goal>
      step attending
  =imaginal> 
  )
 
 (p* lookup-token
   =goal>
     isa encode-task
     next-state =post
     step attending
     role =role
   =visual>
     isa text
     value =val
  =imaginal>
    isa pyramid
  ?imaginal>
       state free
   ==>
!eval! (string-to-chunk =val)
  +retrieval>
    isa equation-token
    string =val
   +goal>
     step dereferencing
     value =val
  +imaginal> =imaginal
 )

 (p* collect-result
   =goal>
     isa encode-task
     next-state =post
     step dereferencing
     role =role
   - role skip
   =retrieval>
     isa equation-token
   =imaginal>
     isa pyramid
  ?imaginal>
    state free
   ==>
   +goal>
     step ready
     state =post
     next-state nil
   +imaginal>
      =role =retrieval
 )

 (p* skip-result
   =goal>
     isa encode-task
     next-state =post
     step dereferencing
     role skip
   =retrieval>
     isa equation-token
  ?imaginal>
    state free
   ==>
   +goal>
     step ready
     state =post
     next-state nil
 )
 
;productions that classify

(p try-retrieval
   =goal>
       isa encode-task
       state determine-answer
   =imaginal>
       isa pyramid
       base =base
       height =height
   ?imaginal>
       state free
==>
   =goal>
       state  try-retrieval
   =imaginal>
    +retrieval>
       isa pyramid
       base =base
       height =height
    -  value nil)

(p* use-retrieved-answer
    =goal>
       isa encode-task
       state try-retrieval
   =IMAGINAL>
       ISA PYRAMID
       BASE =BASE
       HEIGHT =HEIGHT
     =retrieval>
       isa pyramid
       value =val
 ==>
    +visual>
      isa move-attention
      screen-pos base-position
   =GOAL>
       STATE judge-retrieval
       value =val
   =IMAGINAL>
   +METACOGNITIVE>
       ISA PYRAMID
       BASE =BASE
       HEIGHT =HEIGHT
       value =val
)

(P* START-OUTPUT4
   =GOAL>
       ISA ENCODE-TASK
       STATE judge-RETRIEVAL
       value =val
   =metacognitive>
       ISA classify-problem
       type retrieve
    ==>
  !eval! (log-experiment-event 3)
  !eval! (log-experiment-event 4)
  !eval! (log-experiment-event 5)
    +manual>
       ISA         press-key
       key         "N"
    +GOAL>
       isa respond-task
       STATE ANSWER
       STEP START
       VALUE =val
       pos 0)

(p select-strategy
   =goal>
       isa encode-task
       state try-retrieval
   =imaginal>
       isa pyramid
       base =base
       height =height
   ?retrieval>
     state       error
==>
   =goal>
       state classify-problem
   =imaginal>
    +metacognitive>
       isa pyramid
       base =base
       height =height)


(P BEGIN-SOLVING-pyramid1
   =GOAL>
       ISA encode-TASK
       STATE classify-problem
   =metacognitive>
       ISA classify-problem
       type compute
==>
  !eval!  (log-experiment-event 3)
  !eval!  (log-experiment-event 4)
   +goal>
      isa instruction-task
      state start-pyramid
      step ready)

(P BEGIN-SOLVING-pyramid2
   =GOAL>
       ISA encode-TASK
       STATE classify-problem
   =metacognitive>
       ISA classify-problem
       type retrieve
    =imaginal>
       isa pyramid
       base =base
       height =height
==>
  =imaginal>
  !eval!  (log-experiment-event 3)
  !eval!  (log-experiment-event 4)
   +goal>
      isa instruction-task
      state retrieve-answer
   +retrieval>
       isa pyramid
       base =base
       height =height
     - value nil)


(p find-operator
    =goal>
       isa instruction-task
       state =state
       step ready
     - post guided
     - post prepare
==>
    +goal>
       step retrieving-operator
    +retrieval>
      isa operator
      pre =state)

;the productions that know how to do specific things



(p* variable-set
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action variable-set
            arg1 =var1
            arg2 =var2
            post =post
    =imaginal>
        isa pyramid
        =var2 =val
   ?imaginal>
      state free
    ==>
    =imaginal>
       =var1 =val
    +goal>
       step ready
       state =post)

(p* constant-set
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action constant-set
            arg1 =var1
            arg2 =val
            post =post
    =imaginal>
        isa pyramid
   ?imaginal>
      state free
    ==>
    =imaginal>
       =var1 =val
    +goal>
       step ready
       state =post)


(p set-two
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action count-set
            arg1 n2
            post =post
 ==> 
   +goal>
       step ready
       state =post
   +MANUAL>
       ISA HOLD-KEY
       HAND LEFT
       FINGER index
!eval! (setf *finger-count* 2))

(p increment-counter
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action count-increment
            post =post
 ==> 
!bind! =finger (next-finger)
!eval! (incf *finger-count*)
   +goal>
       step ready
       state =post
   +MANUAL>
       ISA HOLD-KEY
       HAND LEFT
       FINGER =finger)



(p* fail-test-count
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action test-equal
            arg1 count
            arg2 =var2
            fail =post
   =IMAGINAL>
       ISA PYRAMID
       =var2 =val2
!eval! (not (equal (number-to-chunk *finger-count*) =val2))
    ==>
   =IMAGINAL>
    +goal>
       step ready
       state =post)



(p* succeed-test-count
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action test-equal
            arg1 count
            arg2 =var2
            success =post
   =IMAGINAL>
       ISA PYRAMID
      =var2 =val2
!eval! (equal (number-to-chunk *finger-count*) =val2)
    ==>
   =IMAGINAL>
    +goal>
       step ready
       state =post)




(p* decrement
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action decrement
            arg1 =var
            post =post
   =IMAGINAL>
       ISA PYRAMID
       =var =x1
    ==>
    +retrieval>
        isa fact
        relation - 
        arg1 =x1
        arg2 n1
   =imaginal>
    +goal>
       step retrieving-fact
        destination =var
        relation - 
        arg1 =x1
        arg2 n1 
       state =post)

(p* increment
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action increment
            arg1 =var
            post =post
   =IMAGINAL>
       ISA PYRAMID
       =var =x1
!eval!   (digitp =x1)
    ==>
    +retrieval>
        isa fact
        relation + 
        arg1 =x1
        arg2 n1
   =imaginal>
    +goal>
       step retrieving-fact
        destination =var
        relation +
        arg1 =x1
        arg2 n1 
       state =post)



(p* subtract-value
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action subtract
            arg1 =var1
            arg2 =val2
            result =dest
            post =post
     =imaginal>
           isa pyramid
           =var1 =val1
   ?imaginal>
      state free
    ==>
    =imaginal>
    =goal>
       step retrieving-fact
       state =post
       destination =dest
    +retrieval>
      isa fact
      arg1 =val1
      arg2 =val2
      relation -)


;subgoal multiplication
(p* multiply
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action multiply
            arg1 =var1
            arg2 =var2
            result =dest
            post =post
     =imaginal>
           isa pyramid
           =var1 =val1
           =var2 =val2
   ?imaginal>
      state free
    ==>
    =goal>
       step multiplying
       state =post
       destination =dest
       problem =imaginal
    +goal>
        isa multiplication-task
        step split-first
       parent =goal
     +imaginal>
        isa multiplication-problem
        number1 =val1
        number2 =val2)


(p returning-from-multiply
    =goal>
       isa instruction-task
       step multiplying
       problem =problem
     =imaginal>
           isa multiplication-problem
           answer =answer
    ==>
    =goal>
       step reinstating
       result =answer
    +retrieval> =problem)

(p single-digits-multiply
     =goal>
        isa multiplication-task
        step split-first
   ?imaginal>
      state free
   =imaginal>
      isa multiplication-problem
      number1 =n1
      number2 =n2
!eval! (digitp =n1)
!eval! (digitp =n2)
==>
    =imaginal>
    +retrieval>
        isa fact
        relation *
        arg1 =n1
        arg2 =n2
     =goal>
        step done)

(p fraction-multiply
     =goal>
        isa multiplication-task
        step split-first
   ?imaginal>
      state free
   =imaginal>
      isa multiplication-problem
      number1 =n1
      number2 =n2
!eval! (fractionp =n1)
==>
!bind! =denom (extract-denominator =n1)
!eval!  (list =n2 =denom)
     =imaginal>
     =goal>
        step done
     +retrieval> 
          isa fact
          arg1 =n2
          arg2 =denom
          relation /)

(p split-first-multiply
     =goal>
        isa multiplication-task
        step split-first
   ?imaginal>
      state free
   =imaginal>
      isa multiplication-problem
      number1 =n1
      number2 =n2
!eval! (multi-digitp =n1)
!eval! (digitp =n2)
==>
     =imaginal>
     =goal>
        step splitting-first
     +retrieval> =n1
       )

(p splitting-first-zero
     =goal>
        isa multiplication-task
        step splitting-first
     =retrieval>
            isa equation-token
            tens =tens
            ones n0
   =imaginal> 
      isa multiplication-problem
      number2 =n2
==>
     =goal>
        step multiply-round
     +retrieval>
        isa fact
        arg1 =tens
        arg2 =n2
        relation *
     =imaginal>)

(p compose-answer
     =goal>
        isa multiplication-task
        step multiply-round
        parent =parent
     =retrieval>
            isa fact
            arg3 =prod
   =imaginal> 
      isa multiplication-problem
      number2 =n2
==>
!bind! =ans (compose-number =prod 'n0)
   +IMAGINAL>
       ANSWER =ANS
   +GOAL> =PARENT
)


(p splitting-first-non-zero
     =goal>
        isa multiplication-task
        step splitting-first
     =retrieval>
            isa equation-token
            tens =tens
            ones =ones
          - ones n0
   =imaginal> 
      isa multiplication-problem
      number2 =n2
==>
     =goal> 
         step mutlipy-ones
     +imaginal>
        tens1 =tens
        ones1 =ones
     +retrieval>
        isa fact
        arg1 =ones
        arg2 =n2
        relation *)

(p split-answer-ones
     =goal>
        isa multiplication-task
        step mutlipy-ones
     =retrieval>
            isa fact
            arg3 =prod
==>
     =goal> 
         step splitting-answer-ones
     +retrieval> =prod)

(p store-answer-ones
     =goal> 
         isa multiplication-task
         step splitting-answer-ones
     =retrieval>
         isa equation-token
         ones =ones
         tens =tens
      =imaginal>
         isa multiplication-problem
         tens1 =tens1
         number2 =n2
==> 
     =goal>
       step MULTIPY-TENS
     +imaginal>
        onesa =ones
        tensa =tens
      +retrieval>
        isa fact
        arg1 =tens1
        arg2 =n2
        relation *)

(p split-answer-tens
     =goal>
        isa multiplication-task
        step MULTIPY-TENS
     =retrieval>
            isa fact
            arg3 =prod
     =imaginal>
            isa multiplication-problem
            tensa =tens
!eval! (digitp =prod)
==>
     =goal> 
         step adding-tens
     =imaginal>
      +retrieval>
        isa fact
        arg1 =tens
        arg2 =prod
        relation +)
  
(p compose-two-digit
     =goal>
        isa multiplication-task
        step adding-tens
        parent =parent
     =retrieval>
        isa fact     
        arg3 =sum
     =imaginal>
         isa multiplication-problem
         onesa =ones
==>  
!bind! =ans (compose-number =sum =ones)
   +IMAGINAL>
       ANSWER =ANS
   +GOAL> =PARENT)

(spp compose-two-digit :at 2)




;subgoaling addition
(p* add
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action add
            arg1 =var1
            arg2 =var2
            result =dest
            post =post
     =imaginal>
           isa pyramid
           =var1 =val1
           =var2 =val2
   ?imaginal>
      state free
    ==>
    =goal>
       step adding
       state =post
       destination =dest
       problem =imaginal
    +goal>
       isa addition-task
       step split-first
       parent =goal
     +imaginal>
        isa addition-problem
        number1 =val1
        number2 =val2)

(p* add-constant
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action add
            arg1 =var1
            arg2 =val2
            result =dest
            post =post
     =imaginal>
           isa pyramid
           =var1 =val1
!eval! (digitp =val2)
    ==>
    =goal>
       step adding
       state =post
       destination =dest
       problem =imaginal
    +goal>
       isa addition-task
       step split-first
       parent =goal
     +imaginal>
        isa addition-problem
        number1 =val1
        number2 =val2)

(p split-first-single
     =goal>
        isa addition-task
        step split-first
   ?imaginal>
      state free
   =imaginal>
      isa addition-problem
      number1 =n1
      number2 =n2
!eval! (digitp =n1)
!eval! (multi-digitp =n2)
==>
     =goal>
        step split-second
     +imaginal>
        tens1 n0
        ones1 =n1)

(p single-digits
     =goal>
        isa addition-task
        step split-first
   ?imaginal>
      state free
   =imaginal>
      isa addition-problem
      number1 =n1
      number2 =n2
!eval! (digitp =n1)
!eval! (digitp =n2)
==>
    =imaginal>
    +retrieval>
        isa fact
        relation +
        arg1 =n1
        arg2 =n2
     =goal>
        step done)

(p split-first-multi
     =goal>
        isa addition-task
        step split-first
   ?imaginal>
      state free
   =imaginal>
      isa addition-problem
      number1 =n1
!eval! (multi-digitp =n1)
==>
     =imaginal>
     =goal>
        step splitting-first
     +retrieval> =n1
       )

(p splitting-first
     =goal>
        isa addition-task
        step splitting-first
     =retrieval>
            isa equation-token
            ones =ones
            tens =tens
   =imaginal> 
      isa addition-problem
==>
     =goal>
        step split-second
     +imaginal>
        tens1 =tens
        ones1 =ones)

(p split-second-single
     =goal>
        isa addition-task
        step split-second
   ?imaginal>
      state free
   =imaginal>
      isa addition-problem
      number2 =n2
!eval! (digitp =n2)
==>
     =goal>
        step add-ones
     +imaginal>
        tens2 n0
        ones2 =n2)

(p split-second-multi
     =goal>
        isa addition-task
        step split-second
   ?imaginal>
      state free
   =imaginal>
      isa addition-problem
      number2 =n2
!eval! (multi-digitp =n2)
==>
     =imaginal>
     =goal>
        step splitting-second
     +retrieval> =n2
       )

(p splitting-second
     =goal>
        isa addition-task
        step splitting-second
     =retrieval>
            isa equation-token
            ones =ones
            tens =tens
   =imaginal> 
      isa addition-problem
==>
     =goal>
        step add-ones
     +imaginal>
        tens2 =tens
        ones2 =ones)

(p add-ones
     =goal>
        isa addition-task
        step add-ones
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      ones1 =ones1
   -  ones1 n0
      ones2 =ones2
   -  ones2 n0
==>
     =goal>
        step adding-ones
     =imaginal>
    +retrieval>
        isa fact
        relation +
        arg1 =ones1
        arg2 =ones2)

(p add-ones-zero1
     =goal>
        isa addition-task
        step add-ones
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      ones1 n0
      ones2 =ones2
==>
     +imaginal>
        onesa =ones2
     =goal>
        step add-tens)

(p add-ones-zero2
     =goal>
        isa addition-task
        step add-ones
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      ones1 =ones1
      ones2 n0
==>
     +imaginal>
        onesa =ones1
     =goal>
        step add-tens)

(p add-ones-multi
    =goal>
       isa addition-task
       step adding-ones
     =retrieval>
            isa fact
            arg3 =val
!eval! (multi-digitp =val) 
==>
    +retrieval> =val
    +goal>
       step splitting-ones-sum)

(p splitting-ones-sum
    =goal>
       isa addition-task
       step splitting-ones-sum
     =retrieval>
            isa equation-token
            ones =ones
            tens n1
     =imaginal>
           isa addition-problem
   ?imaginal>
      state free
==>
    +imaginal>
        onesa =ones
        carry n1
    +goal>
       step add-tens)

(p add-ones-single
    =goal>
       isa addition-task
       step adding-ones
     =retrieval>
            isa fact
            arg3 =val
     =imaginal>
           isa addition-problem
   ?imaginal>
      state free
!eval! (digitp =val) 
==>
    +imaginal>
        onesa =val
    +goal>
       step add-tens)

(p add-tens-zero1
     =goal>
        isa addition-task
        step add-tens
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      tens1 n0
      tens2 =tens2
==>
     +imaginal>
        tensa =tens2
     =goal>
        step check-carry)

(p add-tens-zero2
     =goal>
        isa addition-task
        step add-tens
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      tens1 =tens1
      tens2 n0
==>
     +imaginal>
        tensa =tens1
     =goal>
        step check-carry)

(p add-tens
     =goal>
        isa addition-task
        step add-tens
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      tens1 =tens1
    - tens1 n0
      tens2 =tens2
    - tens2 n0
==>
     =goal>
        step adding-tens
     =imaginal>
    +retrieval>
        isa fact
        relation +
        arg1 =tens1
        arg2 =tens2)

(p addding-tens
     =goal>
        isa addition-task
        step adding-tens
   ?imaginal>
      state free
     =retrieval>
        isa fact
        arg3 =tens
     =imaginal>
        isa addition-problem
==>
     =goal>
        step check-carry
     +imaginal>
       tensa =tens)

(p carry
     =goal>
        isa addition-task
        step check-carry
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      carry =carry
      tensa =tensa
==>
    +retrieval>
        isa fact
        relation +
        arg1 =tensa
        arg2 =carry
     =goal>
        step perform-carry
     =imaginal>)

(p perform-carry
     =goal>
        isa addition-task
        step perform-carry
   =retrieval>
        isa fact
        arg3 =tens
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      onesa =onesa
==>
    =imaginal>
    +retrieval>
        isa equation-token
        ones =onesa
        tens =tens
     =goal>
        step done)

(p no-carry
     =goal>
        isa addition-task
        step check-carry
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      carry nil
      tensa =tensa
==>
    =imaginal>
     =goal>
        step answer)


(P RETURNING-FROM-ADD1
   =GOAL>
       ISA INSTRUCTION-TASK
       STEP ADDING-tens
       problem =problem
   =IMAGINAL>
       ISA ADDITION-PROBLEM
       ANSWER =ANSWER
      NUMBER1 =NUM1
       NUMBER2 =NUM2
 ==>
   =imaginal>
   =GOAL>
       STEP REINSTATING-tens
   +RETRIEVAL> =problem
)

(P REINSTATING2
   =GOAL>
       ISA INSTRUCTION-TASK
       STEP REINSTATING-tens
   =RETRIEVAL>
       ISA PYRAMID
       extra =hundreds
  =imaginal>
       isa addition-problem
       answer =result
!BIND! =ANS2 (COMPOSE-NUMBER =hundreds =result)
 ==>
!BIND! =ANS1 (COMPOSE-NUMBER =hundreds =result)
   =RETRIEVAL>
       result =ans1
   =GOAL>
       STEP ready
   +IMAGINAL> =RETRIEVAL
)

(p answer
     =goal>
        isa addition-task
        step answer
   ?imaginal>
      state free
   =imaginal> 
      isa addition-problem
      tensa =tensa
      onesa =onesa
==>
    =imaginal>
    +retrieval>
        isa equation-token
        ones =onesa
        tens =tensa
     =goal>
        step done)

(p done1
    =goal>
       isa addition-task
       step done
       parent =parent
   =retrieval>
       isa fact
       arg3 =ans
   ?imaginal>
      state free
   =imaginal>
       isa addition-problem
==>
   +imaginal>
       answer =ans
     +goal> =parent)

(p done2
    =goal>
       isa addition-task
       step done
       parent =parent
   =retrieval>
       isa equation-token
   ?imaginal>
      state free
   =imaginal>
       isa addition-problem
==>
   +imaginal>
       answer =retrieval
     +goal> =parent)

(p done3
    =goal>
       isa multiplication-task
       step done
       parent =parent
   =retrieval>
       isa fact
       arg3 =ans
   ?imaginal>
      state free
   =imaginal>
       isa MULTIPLICATION-PROBLEM
==>
   +imaginal>
       answer =ans
     +goal> =parent)

(p* returning-from-add
    =goal>
       isa instruction-task
       step adding
       problem =problem
     =imaginal>
           isa addition-problem
           answer =answer
           number1 =num1
           number2 =num2
!eval! (not (negativep =num1))
    ==>
    =goal>
       step reinstating
       result =answer
    +retrieval> =problem)

(p returning-from-add2
    =goal>
       isa instruction-task
       step adding
       problem =problem
     =imaginal>
           isa addition-problem
           answer =answer
           number1 =num1
           number2 =num2
!eval! (negativep =num1)
    ==>
    =goal>
       step reinstating
       result =answer
    +retrieval> =problem)

(p* reinstating1
    =goal>
       isa instruction-task
       step reinstating
       result =result
      problem =problem
       destination =dest
     =retrieval>
           isa pyramid
    ==>
    =retrieval>
         =dest =result
    =goal>
       step ready
    +imaginal> =retrieval)

(p* harvest-retrieval
    =goal>
       isa instruction-task
       step retrieving-fact
       destination =slot
     =retrieval>
            isa fact
            arg3 =val
     =imaginal>
         isa pyramid
==>
    =imaginal>
       =slot =val
    +goal>
       step ready)

;responding



(p* start-output1
    =goal>
       isa instruction-task
       step retrieving-operator
     =retrieval>
            isa operator
            action output
            arg1 =var1
            post =post
    =imaginal>
        isa pyramid
        =var1 =val
   ?imaginal>
      state free
    ==>
    +visual>
      isa move-attention
      screen-pos base-position
;  !eval! (princ =val)
  !eval! (if (= (length *experiment-events*) 3) (log-experiment-event 4))
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
       pos 0)

(p* start-output2
    =goal>
       isa instruction-task
       state retrieve-answer
     =retrieval>
            isa pyramid
            value =val
    =imaginal>
        isa pyramid
   ?imaginal>
      state free
    ==>
    +visual>
      isa move-attention
      screen-pos base-position
;  !eval! (princ =val)
  !eval! (if (= (length *experiment-events*) 3) (log-experiment-event 4))
  !eval! (log-experiment-event 5)
    +imaginal>
        value =val
    +manual>
       ISA         press-key
       key         "N"
    +GOAL>
       isa respond-task
       STATE ANSWER
       STEP START
       VALUE =val
       pos 0)

 
 (p respond
     =goal>
       isa respond-task
       state answer
       step start
       value =val
   ==>
   !safe-bind! =stringval (chunk-to-string =val)
     +goal>
       step respond ; press-keys
       value =stringval
       pos 0
   )

 (p key-next-digit
    =goal>
      isa respond-task
      step respond
      value =val
      pos =pos
      type nil
    ?manual>
      state free
!eval! (not  (equal (length =val) =pos))
!eval! (not (equal (string (aref =val =pos)) "-"))
  ==>
    +imaginal>
        isa key-location
    !safe-bind! =key  (string (aref =val =pos))
    !bind! =newpos (1+ =pos)
    +manual>
       ISA         press-key
       key         =key
    +goal>
      pos =newpos
      step check
  )

 (p key-next-digit-3
    =goal>
      isa respond-task
      step respond
      value =val
      pos =pos
      type no-check
    ?manual>
      state free
!eval! (not  (equal (length =val) =pos))
!eval! (not (equal (string (aref =val =pos)) "-"))
  ==>
    +imaginal>
        isa key-location
    !safe-bind! =key  (string (aref =val =pos))
    !bind! =newpos (1+ =pos)
    +manual>
       ISA         press-key
       key         =key
    +goal>
      pos =newpos
      step respond
  )

 (p key-next-minus
    =goal>
      isa respond-task
      step respond
      value =val
      pos =pos
    ?manual>
      state free
!eval! (not  (equal (length =val) =pos))
!eval! (equal (string (aref =val =pos)) "-")
==>
    +goal>
      step find-negative
    +retrieval> minus-sign)

 (p find-negative
    =goal>
      isa respond-task
      step find-negative
      pos =pos
      type nil
    =retrieval>
     isa equation-token
  ==>
    +imaginal>
        isa key-location
    !bind! =newpos (1+ =pos)
    +manual>
       ISA         press-key
       key         "-"
    +goal>
      pos =newpos
      step clear
  )

 (p check-motor
    =goal>
      isa respond-task
      step check
    ?manual>
      state free
  ==>
   +visual-location>
      isa visual-location
      :attended new
    +goal>
      step look-at-output
  )

 (p look-at-output
    =goal>
      isa respond-task
      step look-at-output
    =visual-location>
      isa visual-location
==>
    =visual-location>
    +visual>
      isa move-attention
      screen-pos =visual-location
    +goal>
      step read-output
  )

 (p read-output
    =goal>
      isa respond-task
      step read-output
   =VISUAL>
       ISA TEXT
==>
    +goal>
      step respond
  )


 (p clear-motor
    =goal>
      isa respond-task
      step clear
    ?manual>
      state free
  ==>
    +manual>
       ISA         clear
    +goal>
      step respond
  )

 (p key-enter
    =goal>
      isa respond-task
      step respond
      value =val
      pos =pos
    ?manual>
      state free
!eval! (equal (length =val) =pos)
    =visual-location>
      isa visual-location
==>
    +imaginal>
        isa key-location
    +visual>
      isa move-attention
      screen-pos =visual-location
    +manual>
       ISA         press-key
       key         "R"
    +goal>
      step ready
      state done
  )


(p done
     =goal>
       isa respond-task
       state done
   =visual-location>
    isa visual-location     
   ==>
   +imaginal>
       isa pyramid
  +visual>
     isa clear
   +visual-location>
     isa visual-location
     color green
   +GOAL>
       ISA FEEDBACK-TASK
       STEP FIND-FEEDBACK
)

;process-feedback
 (p find-feedback
  =goal>
    isa feedback-task
    step find-feedback
  =visual> 
    isa text
  ?visual-location>
    state free
  ?visual>
    state free
  ==>
    +visual-location>
      isa visual-location
      screen-y current
      screen-x lowest
      > screen-x current
  )


 (p attend-feedback
    =goal>
     isa feedback-task
     step find-feedback
   =visual-location>
     isa visual-location
     color green
   ?visual>
     state free
  ?manual>
    state free
 ==>
   +visual>
     isa move-attention
     screen-pos =visual-location
   +RETRIEVAL>
       ISA PYRAMID
;      - base nil
;      - height nil
     +goal>
       isa feedback-task
       step ready
       state encode-base
)

(p lookup-pyramid
     =goal>
      isa feedback-task
      state encode-base
      step ready
    =retrieval>
      isa pyramid
;      base =base
;      height =height
   ?imaginal>
      state free
;!eval! (and (digitp =base) (digitp =height))
  ==>
   =retrieval>
   +imaginal> =retrieval
   +GOAL>
       STATE CHECK-ANSWER)
   

 (p prepare-rd
    =goal>
    isa feedback-task
    state check-answer
    =visual-location>
      isa visual-location
      kind text
   =retrieval>
      isa pyramid
 ==>
    +goal>
      step repetition-detection
    +manual>
      isa prepare
      style punch
      hand right
      finger index
    )

  (p attending-rd-click
    =goal>
      isa feedback-task
      step repetition-detection
    =visual-location>
      isa visual-location
    kind text
      !safe-eval! (= *nback-match* 1)
  ==>
!eval! (reset-nback)
   +VISUAL>
       ISA MOVE-ATTENTION
       SCREEN-POS =VISUAL-LOCATION
   +goal>
       step repetition-detection
    +manual>
       ISA         punch
      hand right
      finger index
;    +visual>
;      isa clear
  )

  (p clear-visual
    =goal>
      isa feedback-task
      step repetition-detection
   =visual>
      isa text
  ==>
    +visual>
      isa clear
  )

  (p attending-rd-noclick
    =goal>
      isa feedback-task
      step repetition-detection
    =visual-location>
      isa visual-location
      kind text
      !safe-eval! (= *nback-match* 0)
  ==>
!eval! (reset-nback)
   +goal>
       step repetition-detection
  )




(goal-focus goal)
)
