"""设置页面"""

import reflex as rx
from pixiu.config import config


class SettingsState(rx.State):
    """设置状态"""
    glm_api_key: str = ""
    initial_capital: str = "100000"
    commission_rate: str = "0.0003"
    
    def load_settings(self):
        """加载设置"""
        self.glm_api_key = config.glm_api_key
        self.initial_capital = str(config.initial_capital)
        self.commission_rate = str(config.commission_rate)
    
    def save_settings(self):
        """保存设置"""
        import os
        os.environ["GLM_API_KEY"] = self.glm_api_key
        config.glm_api_key = self.glm_api_key
        config.initial_capital = float(self.initial_capital)
        config.commission_rate = float(self.commission_rate)


def page() -> rx.Component:
    """设置页面"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.heading("⚙️ 设置", size="lg"),
                rx.spacer(),
                rx.button("返回", on_click=rx.redirect("/")),
                width="100%",
                margin_bottom="2rem",
            ),
            
            rx.box(
                rx.heading("GLM API 配置", size="md", margin_bottom="1rem"),
                rx.input(
                    placeholder="输入 GLM API Key",
                    value=SettingsState.glm_api_key,
                    on_change=SettingsState.set_glm_api_key,
                    type="password",
                    width="100%",
                ),
                rx.text("用于生成AI智能分析报告", font_size="sm", color="gray.500", margin_top="0.5rem"),
                padding="1.5rem",
                bg="white",
                border_radius="md",
                shadow="sm",
                margin_bottom="1rem",
            ),
            
            rx.box(
                rx.heading("回测参数", size="md", margin_bottom="1rem"),
                
                rx.vstack(
                    rx.hstack(
                        rx.text("初始资金:", width="100px"),
                        rx.input(
                            value=SettingsState.initial_capital,
                            on_change=SettingsState.set_initial_capital,
                            width="200px",
                        ),
                    ),
                    rx.hstack(
                        rx.text("手续费率:", width="100px"),
                        rx.input(
                            value=SettingsState.commission_rate,
                            on_change=SettingsState.set_commission_rate,
                            width="200px",
                        ),
                    ),
                    spacing="1rem",
                ),
                
                padding="1.5rem",
                bg="white",
                border_radius="md",
                shadow="sm",
                margin_bottom="1rem",
            ),
            
            rx.button(
                "保存设置",
                on_click=SettingsState.save_settings,
                color_scheme="blue",
                margin_top="1rem",
            ),
            
            rx.spacer(),
            
            width="100%",
            max_width="600px",
            margin="0 auto",
            padding="2rem",
        ),
        min_height="100vh",
        bg="gray.50",
        on_mount=SettingsState.load_settings,
    )
