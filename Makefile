SRC_DIR = src

all: 
	$(MAKE) -C $(SRC_DIR)
	cp $(SRC_DIR)/fmr .

clean: 
	$(MAKE) -C $(SRC_DIR) $@
