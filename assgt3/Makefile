ROLLNO := EE19B121

default:
	@cd Evaluation_script && ./evaluate.sh
	
all: clean default

log:
	@cat Evaluation_script/submit/$(ROLLNO).log

clean:
	@cd Evaluation_script/submit && rm A3 $(ROLLNO).log  output*.txt
	@cd Evaluation_script && rm A3*
