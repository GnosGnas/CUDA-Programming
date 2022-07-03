ROLLNO := EE19B121

default:
	@cd evaluation-script && ./evaluate.sh
	
all: clean default

log:
	@cat evaluation-script/submit/$(ROLLNO).log

clean:
	@cd evaluation-script/submit && rm A2 $(ROLLNO).log  output*.txt
	@cd evaluation-script && rm A2*
