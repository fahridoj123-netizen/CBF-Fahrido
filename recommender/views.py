from django.shortcuts import render, redirect
from django.http import JsonResponse, Http404
from .engine import ContentBasedRecommender

def get_engine():
    return ContentBasedRecommender.get_instance()

def home_view(request):
    engine = get_engine()
    query = request.GET.get('q', '').strip()
    genre = request.GET.get('genre', 'All').strip()
    
    try:
        page = int(request.GET.get('page', 1))
        page = max(1, page)
    except (ValueError, TypeError):
        page = 1

    search_result = engine.search_movies(query=query, genre=genre, page=page, per_page=20)
    featured_movies = engine.get_featured_movies(limit=6)

    context = {
        'movies': search_result['items'],
        'pagination': search_result,
        'genres': engine.genres_list,
        'current_genre': genre,
        'current_query': query,
        'featured_movies': featured_movies,
        'featured_hero': featured_movies[0] if featured_movies else None,
    }
    return render(request, 'home.html', context)

def movie_detail_view(request, movie_id):
    engine = get_engine()
    movie = engine.get_movie(movie_id)
    if not movie:
        raise Http404("Film tidak ditemukan dalam dataset MovieLens.")

    try:
        k = int(request.GET.get('k', 10))
        k = max(1, min(k, 50))
    except (ValueError, TypeError):
        k = 10

    rec_result = engine.get_recommendations(movie_id, k=k)

    context = {
        'movie': movie,
        'recommendations': rec_result['recommendations'] if rec_result else [],
        'precision_at_k': rec_result['precision_at_k'] if rec_result else 0.0,
        'relevant_count': rec_result['relevant_count'] if rec_result else 0,
        'k': k,
    }
    return render(request, 'movie_detail.html', context)

def methodology_view(request):
    """
    Halaman khusus penjelasan matematis & metodologi:
    - TF-IDF Vectorizer
    - Kalkulasi Sudut Vektor Cosine Similarity
    - Precision@K Evaluation Metric
    """
    return render(request, 'methodology.html')

def api_recommend(request, movie_id):
    """
    API JSON realtime untuk mengambil rekomendasi tanpa reload halaman.
    """
    engine = get_engine()
    try:
        k = int(request.GET.get('k', 10))
        k = max(1, min(k, 50))
    except (ValueError, TypeError):
        k = 10

    rec_result = engine.get_recommendations(movie_id, k=k)
    if not rec_result:
        return JsonResponse({'status': 'error', 'message': 'Movie not found'}, status=404)

    return JsonResponse({
        'status': 'success',
        'data': rec_result
    })

def api_search(request):
    """
    API JSON autocomplete untuk live search bar.
    """
    query = request.GET.get('q', '').strip()
    if not query:
        return JsonResponse({'status': 'success', 'results': []})

    engine = get_engine()
    result = engine.search_movies(query=query, page=1, per_page=8)
    return JsonResponse({
        'status': 'success',
        'results': result['items']
    })
