ROLLNO := EE19B121

default:
	@cd EvaluationScript && ./evaluate.sh
	
all: clean default

log:
	@cat EvaluationScript/SUBMIT/$(ROLLNO).log

clean:
	@cd EvaluationScript/SUBMIT && rm $(ROLLNO).log main output.txt

