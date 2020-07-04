from django.http.response import JsonResponse
from django.shortcuts import render


def home(request):
    return render(request, 'nix/home.html')

def identifica_texto(request):

    texto = request.POST.get('texto', '')
    if texto:
        pass
    
    return JsonResponse({'idioma': 'PT-BR'})
