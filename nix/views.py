from django.http.response import JsonResponse
from django.shortcuts import render
from predict import language


def home(request):
    return render(request, 'nix/home.html')

def identifica_texto(request):

    texto = request.GET.get('texto', '')
    if texto:
        idioma = language(texto)
    else:
        idioma = ''
    
    return JsonResponse({'idioma': idioma})
