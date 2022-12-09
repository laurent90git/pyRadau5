CC = gcc
FC = gfortran
LD = gcc

CCFLAGS = -fPIC
FCFLAGS = -fPIC -fno-underscoring
LDFLAGS = -shared

OBJS = dc_decsol.o  decsol.o radau5.o rock4.o integration_radau.o integration_rock.o

LIB = lib_radau_rock.so

all: $(LIB)

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lgfortran

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

%.o : %.f
	$(FC) $(FCFLAGS) -c $< -o $@

clean:
	rm -f $(LIB) $(OBJS)
