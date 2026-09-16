import 'dart:convert';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:partio_gui/api/api_client.dart';
import 'package:partio_gui/api/providers.dart';
import 'package:partio_gui/app.dart';

http.Response _json(Object body) =>
    http.Response(jsonEncode(body), 200, headers: {
      'content-type': 'application/json',
    });

void main() {
  testWidgets('shows the nav shell and an empty library', (tester) async {
    final fakeClient = MockClient((request) async {
      if (request.url.path == '/api/feeds') return _json([]);
      if (request.url.path == '/api/tracks') return _json([]);
      throw StateError('unexpected request: ${request.url}');
    });

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          apiClientProvider.overrideWithValue(
            ApiClient(baseUrl: 'http://backend.invalid', httpClient: fakeClient),
          ),
        ],
        child: const PartioGuiApp(),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Library'), findsWidgets);
    expect(find.text('No feeds remembered yet.'), findsOneWidget);
    expect(find.text('Nothing here yet.'), findsOneWidget);
    expect(find.text('Teach a jingle'), findsOneWidget);
    expect(find.text('Find'), findsOneWidget);
    expect(find.text('Locate'), findsOneWidget);
    expect(find.text('Cut'), findsOneWidget);
  });
}
