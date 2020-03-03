from django.db import models


class SRD(models.Model):
    review = models.TextField()
    predict_label = models.CharField(max_length=100)

    def __str__(self):
        return '{}'.format(self.review)


class FEAD(models.Model):
    description = models.TextField()
    review = models.TextField()
    predict_label = models.CharField(max_length=100)

    def __str__(self):
        return '{}-{}'.format(self.description, self.review)

