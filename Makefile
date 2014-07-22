SRC_DIR = src

all: 
	$(MAKE) -C $(SRC_DIR) build
	$(MAKE) -C $(SRC_DIR) test
	cp $(SRC_DIR)/fmr .

build:
	$(MAKE) -C $(SRC_DIR) build
	
test:
	$(MAKE) -C $(SRC_DIR) test

clean: 
	$(MAKE) -C $(SRC_DIR) $@
