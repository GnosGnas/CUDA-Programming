ROLLNO := EE19B121

default:
	@cd EvaluationScript/SUBMIT/$(ROLLNO) && rm -rf Input/ Output/ main* kernel* $(ROLLNO).log
	@#@mkdir EvaluationScript/SUBMIT/$(ROLLNO)
	@#@cp main.cu EvaluationScript/SUBMIT/$(ROLLNO)/$(ROLLNO).cu
	@chmod +x EvaluationScript/evaluate.sh
	@cd EvaluationScript && ./evaluate.sh


log:
	@cat EvaluationScript/SUBMIT/$(ROLLNO)/$(ROLLNO).log

clean:
	@cd EvaluationScript/SUBMIT && rm -rf *

normal:
	@cd EvaluationScript && ./evaluate.sh
	@cd EvaluationScript && rm A*
