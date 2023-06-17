"""djangochat URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

import app.views

urlpatterns = [
    path('', app.views.to_chatgpt2),
    path('admin/', admin.site.urls),
    path('chatgpt2/', app.views.chatgpt2_view),
    path('blenderbot/', app.views.blenderbot_view),
    path('chatglm/', app.views.chatglm_view),
    path('to_chatglm/', app.views.to_chatglm),
    path('to_chatgpt2/', app.views.to_chatgpt2),
    path('to_blender/', app.views.to_blenderbot),
    path('to_load_chatglm/', app.views.to_load_chatglm),
]
