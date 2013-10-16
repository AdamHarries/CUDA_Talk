CPP_FILES = $(wildcard *.md)
OBJ_FILES = $(patsubst %.md,%.pdf,$(CPP_FILES))

%.pdf: %.md
	pandoc -t beamer --slide-level=2  cuda_talk.md -o cuda_talk.pdf
	pandoc talk_notes.md -o cuda_notes.pdf

all: $(OBJ_FILES)

clean:
	rm -f *.pdf
