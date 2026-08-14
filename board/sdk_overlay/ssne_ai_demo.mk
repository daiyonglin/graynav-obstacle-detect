# GrayNav unified single-model Buildroot package contract.
SSNE_AI_DEMO_VERSION =
SSNE_AI_DEMO_SITE = $(S1SRC)/app_demo/$(call qstrip,$(BR2_PACKAGE_SSNE_AI_DEMO_APP)/ssne_ai_demo)
SSNE_AI_DEMO_SITE_METHOD = local

export EXPORT_LIB_M1_SDK_ROOT_PATH = $(call qstrip,$(BR2_M1_SDK_ROOT_PATH))

A1_YOLO_NUM_CLASSES ?= 8
A1_YOLO_INPUT_CHANNELS ?= 1
A1_ENABLE_VOICE ?= ON
A1_REQUIRE_MODEL ?= ON
A1_MODEL_FILENAME ?= graynav_unified_indoor8_scene21.m1model

SSNE_AI_DEMO_CONF_OPTS += \
	-DA1_YOLO_NUM_CLASSES=$(A1_YOLO_NUM_CLASSES) \
	-DA1_YOLO_INPUT_CHANNELS=$(A1_YOLO_INPUT_CHANNELS) \
	-DA1_ENABLE_VOICE=$(A1_ENABLE_VOICE) \
	-DA1_REQUIRE_MODEL=$(A1_REQUIRE_MODEL) \
	-DA1_MODEL_FILENAME=$(A1_MODEL_FILENAME)

define SSNE_AI_DEMO_BUILD_CMDS
	$(MAKE) CC="$(TARGET_CC)" -C $(@D) all
endef

define SSNE_AI_DEMO_INSTALL_TARGET_CMDS
	rm -rf $(TARGET_DIR)/app_demo/
	mkdir -p $(TARGET_DIR)/app_demo/app_assets/models
	mkdir -p $(TARGET_DIR)/app_demo/app_assets/osd
	$(INSTALL) -D -m 0755 $(@D)/ssne_ai_demo $(TARGET_DIR)/app_demo/
	$(INSTALL) -m 0644 $(@D)/app_assets/colorLUT.sscl \
		$(TARGET_DIR)/app_demo/app_assets/colorLUT.sscl
	$(INSTALL) -m 0644 $(@D)/app_assets/models/$(A1_MODEL_FILENAME) \
		$(TARGET_DIR)/app_demo/app_assets/models/$(A1_MODEL_FILENAME)
	for name in STOP SLOW CLEAR LEFT RIGHT; do \
		$(INSTALL) -m 0644 $(@D)/app_assets/osd/$$name.ssbmp \
			$(TARGET_DIR)/app_demo/app_assets/osd/$$name.ssbmp; \
	done
	for asset in $(@D)/app_assets/osd/NAV_*.ssbmp; do \
		test -f $$asset || exit 1; \
		$(INSTALL) -m 0644 $$asset \
			$(TARGET_DIR)/app_demo/app_assets/osd/$$(basename $$asset); \
	done
	cp -r $(@D)/scripts/. $(TARGET_DIR)/app_demo/scripts/
endef

$(eval $(cmake-package))
