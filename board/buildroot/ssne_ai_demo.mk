# Tracked mirror of smart_software/package/ssne_ai_demo/ssne_ai_demo.mk.
SSNE_AI_DEMO_VERSION =
SSNE_AI_DEMO_SITE = $(S1SRC)/app_demo/$(call qstrip,$(BR2_PACKAGE_SSNE_AI_DEMO_APP)/ssne_ai_demo)
SSNE_AI_DEMO_SITE_METHOD = local

export EXPORT_LIB_M1_SDK_ROOT_PATH = $(call qstrip,$(BR2_M1_SDK_ROOT_PATH))

A1_YOLO_NUM_CLASSES ?= 80
A1_YOLO_INPUT_CHANNELS ?= 3
A1_ENABLE_VOICE ?= ON
A1_ENABLE_SURFACE_SEG ?= ON
A1_MODEL_FILENAME ?= yolov8n80_graycopy_head6.m1model
A1_SEG_MODEL_FILENAME ?= graynav_surface_depth_e3_gray1.m1model

SSNE_AI_DEMO_CONF_OPTS += \
	-DA1_YOLO_NUM_CLASSES=$(A1_YOLO_NUM_CLASSES) \
	-DA1_YOLO_INPUT_CHANNELS=$(A1_YOLO_INPUT_CHANNELS) \
	-DA1_ENABLE_VOICE=$(A1_ENABLE_VOICE) \
	-DA1_ENABLE_SURFACE_SEG=$(A1_ENABLE_SURFACE_SEG) \
	-DA1_MODEL_FILENAME=$(A1_MODEL_FILENAME) \
	-DA1_SEG_MODEL_FILENAME=$(A1_SEG_MODEL_FILENAME)

define SSNE_AI_DEMO_BUILD_CMDS
	$(MAKE) CC="$(TARGET_CC)" -C $(@D) all
endef

define SSNE_AI_DEMO_INSTALL_TARGET_CMDS
	rm -rf mkdir $(TARGET_DIR)/app_demo/
	mkdir $(TARGET_DIR)/app_demo/
	$(INSTALL) -D -m 0755 $(@D)/ssne_ai_demo $(TARGET_DIR)/app_demo/
	cp -r $(@D)/app_assets/. $(TARGET_DIR)/app_demo/app_assets/
	find $(TARGET_DIR)/app_demo/app_assets/models -type f -name '*.m1model' \
		! -name '$(A1_MODEL_FILENAME)' ! -name '$(A1_SEG_MODEL_FILENAME)' -delete
	cp -r $(@D)/scripts/. $(TARGET_DIR)/app_demo/scripts/
endef

$(eval $(cmake-package))
