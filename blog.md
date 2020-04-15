---
layout: page
title: Blog
permalink: /blog/
---

{% for post in site.posts %}

<article>
<div style="margin-bottom:16px;padding:24px" class="card">
<h2>
<a style="color:#000000" href="{{ post.url }}">
{{ post.title }}
</a>
</h2>
<time class="time" datetime="{{ post.date | date: "%Y-%m-%d" }}">{{ post.date | date_to_long_string }}</time>
<p class="text">
{{post.excerpt }}
</p>
{% if post.tags.size > 0 %}
<p>
<span>
{% for tag in post.tags %}
<span style="color:#ffffff;background-color:#000000" class="badge badge-secondary">
{{tag}}
</span>
&nbsp;
{% endfor %}
</span>
</p>
{% endif %}
</div>
</article>
{% endfor %}
